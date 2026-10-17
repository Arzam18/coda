//! EPD file loading and test suite runner.

use std::fs;
use std::time::Instant;

use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::*;
use crate::types::*;

/// A single EPD test position.
pub struct EpdPosition {
    pub fen: String,
    pub best_moves: Vec<String>,  // "bm" field: correct moves in SAN or coordinate
    pub avoid_moves: Vec<String>, // "am" field: moves to avoid
    /// "dm" field: a mate in at most this many moves must be found.
    pub direct_mate: Option<i32>,
    pub id: String,
}

impl EpdPosition {
    /// True when the line carries something the runner can grade. Perft
    /// suites (`D4 23441`) and bare position lists carry no solution opcode;
    /// they used to be counted as "trivially passing", which made a whole
    /// file report 100% without a single move being checked.
    pub fn is_gradable(&self) -> bool {
        !self.best_moves.is_empty() || !self.avoid_moves.is_empty() || self.direct_mate.is_some()
    }
}

/// Resolve the NNUE net for an EPD run.
///
/// Mirrors every other subcommand (`bench`, `datagen`, `eval-fens`): an explicit
/// `-n` wins and is fatal if it fails to load, otherwise fall back to the shared
/// `auto_discover_nnue` (embedded net > ./net.nnue > net.txt discovery).
///
/// `epd` used to skip the fallback entirely, so on a `make` build — which HAS a
/// net embedded — `coda epd file.epd` ran with no net at all and panicked deep
/// inside the search ("no NNUE net loaded", search.rs) on the first evaluate.
/// It was the only subcommand missing the fallback.
///
/// Returns Err with a ready-to-print message instead of exiting, so the failure
/// paths are testable.
pub(crate) fn resolve_net(info: &mut SearchInfo, nnue_path: Option<&str>) -> Result<(), String> {
    if let Some(path) = nnue_path {
        // An EXPLICIT override that fails must be fatal: silently falling back
        // to the embedded net produces plausible-looking but wrong suite results.
        return info
            .load_nnue(path)
            .map_err(|e| format!("failed to load NNUE '{}': {}", path, e));
    }
    if !info.auto_discover_nnue() {
        return Err(
            "no NNUE net found (looked for an embedded net, ./net.nnue, and net.txt \
             discovery). Build with `make` to embed one, or pass -n <path>."
                .to_string(),
        );
    }
    Ok(())
}

/// Parse an EPD file into positions.
pub fn parse_epd(path: &str) -> Vec<EpdPosition> {
    let bytes = fs::read(path).expect("Failed to read EPD file");
    let content = String::from_utf8_lossy(&bytes);
    content.lines().filter_map(parse_epd_line).collect()
}

/// Opcodes recognised when an operation is not `;`-terminated, e.g.
/// `... w - - 0 1 bm Rc8+ dm 26`. Only these start a new operation inside a
/// `;`-less run; anything else is an operand of the current one. None of them
/// can be a SAN move (SAN pieces are KQRBN, squares are lowercase) or a UCI
/// move, so a move is never mistaken for an opcode. (`c0`..`c9` can be — see
/// `is_comment_opcode`.)
fn is_opcode(tok: &str) -> bool {
    matches!(
        tok,
        "bm" | "am" | "dm" | "id" | "ce" | "pv" | "pm" | "sm" | "acd" | "acn" | "acs"
            | "hmvc" | "fmvn" | "draw_reject" | "noop" | "eco" | "nic" | "tcgs" | "tcri" | "tcsi"
    ) || is_numbered_opcode(tok, 'D')
}

/// Comment opcodes `c0`..`c9` collide with pawn moves to the c-file (`bm c5`),
/// so inside a `;`-less run they only open an operation when a quoted comment
/// follows, which is how comments are always written.
fn is_comment_opcode(tok: &str, next: Option<&String>) -> bool {
    is_numbered_opcode(tok, 'c') && next.map_or(false, |n| n.starts_with('"'))
}

/// `c0`..`c9`, `v0`..`v9`, and perft depths `D1`, `D12`, ...
fn is_numbered_opcode(tok: &str, prefix: char) -> bool {
    tok.strip_prefix(prefix)
        .map_or(false, |n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Split the text after the position into `(opcode, operands)` pairs.
///
/// Standard EPD terminates each operation with `;`, but real suites also
/// omit the semicolons, put a stray `;` straight after the FEN, quote ids,
/// and annotate moves (`Qd1 +`, `Nxb6 + ?`). Quoted strings are kept whole
/// so a `;` inside an id or comment doesn't split it.
fn split_operations(ops: &str) -> Vec<(String, Vec<String>)> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for ch in ops.chars() {
        match ch {
            '"' => { in_quote = !in_quote; cur.push(ch); }
            ';' if !in_quote => {
                if !cur.is_empty() { tokens.push(std::mem::take(&mut cur)); }
                tokens.push(";".to_string());
            }
            c if c.is_whitespace() && !in_quote => {
                if !cur.is_empty() { tokens.push(std::mem::take(&mut cur)); }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() { tokens.push(cur); }

    let mut result: Vec<(String, Vec<String>)> = Vec::new();
    // After a `;` the next token is an opcode by definition; inside a
    // `;`-less run only a recognised opcode opens a new operation.
    let mut expect_opcode = true;
    for (i, tok) in tokens.iter().enumerate() {
        if tok == ";" {
            expect_opcode = true;
            continue;
        }
        if expect_opcode || is_opcode(tok) || is_comment_opcode(tok, tokens.get(i + 1)) || result.is_empty() {
            result.push((tok.clone(), Vec::new()));
            expect_opcode = false;
        } else if let Some(last) = result.last_mut() {
            last.1.push(tok.clone());
        }
    }
    result
}

/// Normalise a move operand: drop check/mate/annotation suffixes (`Rc8+`,
/// `Qxf7#`, `Nf3!?`). Returns None for operands that are only annotation
/// (`+`, `?`) — some suites write `bm Qd1 +` with the check sign spaced off.
fn clean_move(tok: &str) -> Option<String> {
    let m = tok.trim_end_matches(|c: char| matches!(c, '+' | '#' | '!' | '?'));
    if m.is_empty() { None } else { Some(m.to_string()) }
}

fn parse_epd_line(raw: &str) -> Option<EpdPosition> {
    // Some suites are NUL-padded or CRLF-terminated.
    let line = raw.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let mut fields = line.split_whitespace();
    let first = fields.next()?;
    let (fen_base, rest): (String, String) = if first == "startpos" {
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -".to_string(),
            fields.collect::<Vec<_>>().join(" "),
        )
    } else {
        let side = fields.next()?;
        let castle = fields.next()?;
        // The EP field may carry a glued `;` (`-;`) when the ops start at once.
        let ep_tok = fields.next()?;
        let (ep, ep_tail) = match ep_tok.find(';') {
            Some(i) => (&ep_tok[..i], &ep_tok[i..]),
            None => (ep_tok, ""),
        };
        let mut rest: Vec<&str> = Vec::new();
        if !ep_tail.is_empty() { rest.push(ep_tail); }
        rest.extend(fields);
        (format!("{} {} {} {}", first, side, castle, ep), rest.join(" "))
    };

    // Optional FEN move counters (`0 1`) between the position and the ops.
    let mut rest_tokens: Vec<&str> = rest.split_whitespace().collect();
    let mut halfmove = 0u32;
    let mut fullmove = 1u32;
    if ep_tail_is_empty(&rest_tokens) && rest_tokens.len() >= 2 {
        if let (Ok(h), Ok(f)) = (rest_tokens[0].parse::<u32>(), rest_tokens[1].trim_end_matches(';').parse::<u32>()) {
            halfmove = h;
            fullmove = f;
            let glued_semi = rest_tokens[1].ends_with(';');
            rest_tokens.drain(..2);
            if glued_semi { rest_tokens.insert(0, ";"); }
        }
    }
    let ops_str = rest_tokens.join(" ");

    let mut best_moves = Vec::new();
    let mut avoid_moves = Vec::new();
    let mut direct_mate = None;
    let mut id = String::new();

    for (opcode, operands) in split_operations(&ops_str) {
        match opcode.as_str() {
            "bm" => best_moves.extend(operands.iter().filter_map(|m| clean_move(m))),
            "am" => avoid_moves.extend(operands.iter().filter_map(|m| clean_move(m))),
            "dm" => direct_mate = operands.first().and_then(|n| n.parse::<i32>().ok()).filter(|&n| n > 0),
            "id" => id = operands.join(" ").trim_matches('"').trim().to_string(),
            "hmvc" => if let Some(n) = operands.first().and_then(|n| n.parse().ok()) { halfmove = n },
            "fmvn" => if let Some(n) = operands.first().and_then(|n| n.parse().ok()) { fullmove = n },
            _ => {}
        }
    }

    Some(EpdPosition {
        fen: format!("{} {} {}", fen_base, halfmove, fullmove),
        best_moves,
        avoid_moves,
        direct_mate,
        id,
    })
}

/// The move counters may only be read when the ops haven't already started
/// with a `;` glued to the EP field.
fn ep_tail_is_empty(rest: &[&str]) -> bool {
    rest.first().map_or(true, |t| !t.starts_with(';'))
}

/// Convert a move to SAN notation for comparison with EPD best moves.
pub fn move_to_san(board: &Board, mv: Move) -> String {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let pt = board.piece_type_at(from);

    // Castling
    if flags == FLAG_CASTLE {
        // Kingside castling always lands the king on the g-file, queenside on
        // the c-file (Chess960 too, where `to > from` doesn't tell them apart).
        return if file_of(to) == 6 { "O-O".to_string() } else { "O-O-O".to_string() };
    }

    let mut san = String::new();

    // Piece letter (not for pawns)
    if pt != PAWN {
        san.push(match pt {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            KING => 'K',
            _ => '?',
        });
    }

    // Disambiguation: check if another piece of same type can go to the same square
    if pt != PAWN && pt != KING {
        let legal = generate_legal_moves(board);
        let mut same_piece_to = false;
        let mut same_file = false;
        let mut same_rank = false;

        for i in 0..legal.len {
            let other = legal.get(i);
            if other == mv { continue; }
            let other_from = move_from(other);
            let other_to = move_to(other);
            if other_to != to { continue; }
            if board.piece_type_at(other_from) != pt { continue; }

            same_piece_to = true;
            if file_of(other_from) == file_of(from) { same_file = true; }
            if rank_of(other_from) == rank_of(from) { same_rank = true; }
        }

        if same_piece_to {
            if !same_file {
                san.push((b'a' + file_of(from)) as char);
            } else if !same_rank {
                san.push((b'1' + rank_of(from)) as char);
            } else {
                san.push((b'a' + file_of(from)) as char);
                san.push((b'1' + rank_of(from)) as char);
            }
        }
    }

    // Capture
    let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
    if is_capture {
        if pt == PAWN {
            san.push((b'a' + file_of(from)) as char);
        }
        san.push('x');
    }

    // Destination square
    san.push((b'a' + file_of(to)) as char);
    san.push((b'1' + rank_of(to)) as char);

    // Promotion
    if is_promotion(mv) {
        san.push('=');
        san.push(match promotion_piece_type(mv) {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            _ => '?',
        });
    }

    // Check/checkmate suffix
    let mut board_copy = board.clone();
    if !board_copy.make_move(mv) { return san; }
    if board_copy.in_check() {
        // Check if it's checkmate
        let legal_after = generate_legal_moves(&board_copy);
        if legal_after.len == 0 {
            san.push('#');
        } else {
            san.push('+');
        }
    }

    san
}

/// Run an EPD test suite.
pub fn run_epd(path: &str, time_per_pos: u64, max_positions: usize, nnue_path: Option<&str>) {
    let all = parse_epd(path);
    let gradable: Vec<&EpdPosition> = all.iter().filter(|p| p.is_gradable()).collect();
    let ungradable = all.len() - gradable.len();
    let total = if max_positions > 0 { max_positions.min(gradable.len()) } else { gradable.len() };

    println!("Running {} positions from {}", total, path);
    if ungradable > 0 {
        println!("Skipping {} line(s) with no bm/am/dm operation (nothing to grade)", ungradable);
    }
    if total == 0 {
        println!("\nNo gradable positions: this file has no bm/am/dm operations \
                  (a perft or position list?).");
        return;
    }
    println!("Time per position: {}ms", time_per_pos);
    println!();
    // One summary line per position instead of every iteration's info line.

    let mut info = SearchInfo::new(64);
    if let Err(e) = resolve_net(&mut info, nnue_path) {
        eprintln!("FATAL: {}", e);
        std::process::exit(2);
    }
    info.silent = true;
    let mut passed = 0;
    let mut failed = 0;
    let suite_start = Instant::now();

    for (i, pos) in gradable.iter().take(total).enumerate() {
        let mut board = Board::from_fen(&pos.fen);
        info.nodes = 0;
        // `search()` deliberately does NOT reset global_nodes (see its comment:
        // callers own it, so SMP helper contributions aren't clobbered). epd
        // calls `search()` directly and single-threaded, so it must do it here
        // or the counter accumulates across the whole suite: every info line
        // after the first reports the running SUITE total as `nodes`, and
        // divides it by the per-position `time`, so `nps` climbs without bound
        // (63M nps by mid-suite on a ~900k nps engine).
        //
        // Second, quieter consequence: global_nodes is also what the max_nodes
        // limit tests against, so a node-limited epd run would have every
        // position after the first trip its budget immediately.
        //
        // `bench` already carries the identical fix (P2.8); epd was missed.
        // (last_flushed_nodes is reset inside search() itself, so only the
        // shared counter needs clearing here.)
        info.global_nodes.store(0, std::sync::atomic::Ordering::Relaxed);
        info.stop.store(false, std::sync::atomic::Ordering::Relaxed);
        reset_epd_position_state(&mut info);

        let limits = SearchLimits {
            movetime: time_per_pos,
            ..SearchLimits::new()
        };

        info.iter_log = Some(Vec::new());
        let best = search(&mut board, &mut info, &limits);
        let log = info.iter_log.take().unwrap_or_default();
        let score = info.last_score;
        let best_san = move_to_san(&board, best);
        let best_uci = move_to_uci(best);

        let verdict = grade(pos, &best_san, &best_uci, mate_in_moves(score));

        // "Found at": the first iteration from which every later iteration
        // would also have passed, i.e. when the answer settled — not the first
        // lucky hit that a deeper iteration then abandoned.
        let iter_ok = |r: &IterRecord| {
            let san = move_to_san(&board, r.best_move);
            grade(pos, &san, &move_to_uci(r.best_move), mate_in_moves(r.score)).is_ok()
        };
        let settled_from = log.iter().rposition(|r| !iter_ok(r)).map_or(0, |i| i + 1);
        let last = log.last();
        let label = if pos.id.is_empty() { format!("#{}", i + 1) } else { pos.id.clone() };
        let shown = format_score(score);

        match verdict {
            Ok(()) => {
                passed += 1;
                let found = log.get(settled_from).copied();
                let (d, n, t) = match found {
                    Some(r) => (r.depth, r.nodes, r.time_ms),
                    None => (info.completed_depth, info.global_nodes.load(std::sync::atomic::Ordering::Relaxed), 0),
                };
                println!("ok    {}: {} -> {} ({}), found at depth {}, {} nodes, {} ms",
                    label, expectation(pos), best_san, shown, d, fmt_nodes(n), t);
            }
            Err(reason) => {
                failed += 1;
                let (d, n) = last.map_or((info.completed_depth, 0), |r| (r.depth, r.nodes));
                println!("FAIL  {}: {} -> {} ({}), depth {}, {} nodes — {}",
                    label, expectation(pos), best_san, shown, d, fmt_nodes(n), reason);
            }
        }
    }

    let elapsed = suite_start.elapsed();
    println!("\n\nResults: {}/{} passed ({:.1}%)",
        passed, passed + failed,
        100.0 * passed as f64 / (passed + failed) as f64);
    println!("Total time: {:.1}s", elapsed.as_secs_f64());
}

/// The operations being tested, as the line states them: `bm Qc7`, `dm 26`.
fn expectation(pos: &EpdPosition) -> String {
    let mut parts = Vec::new();
    if let Some(n) = pos.direct_mate { parts.push(format!("dm {}", n)); }
    if !pos.best_moves.is_empty() { parts.push(format!("bm {}", pos.best_moves.join(" "))); }
    if !pos.avoid_moves.is_empty() { parts.push(format!("am {}", pos.avoid_moves.join(" "))); }
    parts.join(", ")
}

/// Display score as a GUI would see it (`cp 76`, `mate 12`).
fn format_score(score: i32) -> String {
    let s = crate::tt::format_uci_score(score);
    s.strip_prefix("score ").unwrap_or(&s).to_string()
}

fn fmt_nodes(n: u64) -> String {
    if n < 10_000 { n.to_string() } else { format!("{}k", n / 1000) }
}

/// Moves to mate for the side to move, if `score` is a winning mate score.
fn mate_in_moves(score: i32) -> Option<i32> {
    if crate::tt::is_mate_score(score) && score > 0 {
        Some((crate::tt::MATE_SCORE - score + 1) / 2)
    } else {
        None
    }
}

/// Grade one search result against a position's operations.
///
/// - `dm N`: the search must report a mate in at most N moves. Any move that
///   delivers it counts — a mate suite asks for the mate, and many positions
///   have several mating first moves while `bm` lists only one.
/// - otherwise `bm`: the move played must be one of the best moves.
/// - `am`: the move played must not be an avoid move (checked in both cases).
fn grade(pos: &EpdPosition, best_san: &str, best_uci: &str, found_mate_in: Option<i32>) -> Result<(), String> {
    let san_clean = best_san.trim_end_matches(|c: char| matches!(c, '+' | '#'));
    let matches_move = |mv: &str| mv == san_clean || mv == best_uci;

    let mut reasons: Vec<String> = Vec::new();
    if let Some(n) = pos.direct_mate {
        match found_mate_in {
            Some(m) if m <= n => {}
            Some(m) => reasons.push(format!("found mate in {} (dm {})", m, n)),
            None => reasons.push(format!("no mate found (dm {})", n)),
        }
    } else if !pos.best_moves.is_empty() && !pos.best_moves.iter().any(|m| matches_move(m)) {
        reasons.push(format!("not in bm {:?}", pos.best_moves));
    }
    if pos.avoid_moves.iter().any(|m| matches_move(m)) {
        reasons.push(format!("matched am {:?}", pos.avoid_moves));
    }
    if reasons.is_empty() { Ok(()) } else { Err(reasons.join(" and ")) }
}

fn reset_epd_position_state(info: &mut SearchInfo) {
    info.clear_persistent_histories();
    info.tt.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the bug where `coda epd` skipped auto-discovery.
    ///
    /// It was the only subcommand that did not fall back to the shared
    /// `auto_discover_nnue`, so on a `make` build (which embeds a net) it ran
    /// with NO net and panicked inside the search on the first evaluate rather
    /// than reporting anything useful. The contract this locks down: resolution
    /// either loads a net or returns a clean Err — it never leaves `info`
    /// netless while claiming success, and it never panics.
    #[test]
    fn epd_net_resolution_never_leaves_info_netless_on_ok() {
        // Explicit path that cannot load -> clean Err, not a panic.
        let mut info = SearchInfo::new(1);
        let err = resolve_net(&mut info, Some("/nonexistent/definitely-not-a.nnue"));
        assert!(err.is_err(), "a bad explicit -n must be an error");
        assert!(info.nnue_net.is_none(), "failed load must not leave a net behind");

        // No explicit path -> MUST attempt auto-discovery. Whether a net is
        // found depends on the environment, so assert the invariant rather than
        // the outcome: Ok implies a net really is loaded (the old code could
        // reach the search with none), and Err carries a message.
        let mut info2 = SearchInfo::new(1);
        match resolve_net(&mut info2, None) {
            Ok(()) => assert!(
                info2.nnue_net.is_some(),
                "resolve_net returned Ok without actually loading a net"
            ),
            Err(msg) => assert!(!msg.is_empty(), "Err must explain what to do"),
        }
    }

    #[test]
    fn epd_position_reset_clears_persistent_histories_and_tt() {
        let mut info = SearchInfo::new(1);
        info.dirty_persistent_histories_for_test();

        let hash = 0x1234_5678_9abc_def0;
        info.tt.store(hash, 1, 23, crate::tt::TT_FLAG_EXACT, NO_MOVE, 10, false);
        assert!(info.tt.probe(hash).hit, "test setup must seed TT entry");

        reset_epd_position_state(&mut info);

        info.assert_persistent_histories_clear_for_test();
        assert!(!info.tt.probe(hash).hit, "EPD reset must clear TT between positions");
    }

    fn parse(line: &str) -> EpdPosition {
        parse_epd_line(line).expect("line should parse")
    }

    /// The formats below are all taken from real public suites. Each one used
    /// to be mis-parsed into an empty `bm`, which the runner then counted as a
    /// pass, so whole files reported 100% without a move being checked.
    #[test]
    fn epd_parses_full_fen_with_unterminated_ops() {
        // Move counters present and no `;` between operations.
        let p = parse("7R/2k1p2P/r2rP3/8/3p1R2/8/2P2p2/2K5 w - - 0 1 bm Rc8+ dm 26");
        assert_eq!(p.fen, "7R/2k1p2P/r2rP3/8/3p1R2/8/2P2p2/2K5 w - - 0 1");
        assert_eq!(p.best_moves, vec!["Rc8"]);
        assert_eq!(p.direct_mate, Some(26));

        let p = parse("5N1r/5n1n/ppp3R1/5K2/7k/6p1/6PN/8 w - - 0 1 bm g6g4");
        assert_eq!(p.best_moves, vec!["g6g4"]);

        let p = parse("kqq5/qq6/q7/8/8/5RRR/5RRR/5RRK w - - 0 143 bm Rf8");
        assert!(p.fen.ends_with(" 0 143"));
        assert_eq!(p.best_moves, vec!["Rf8"]);
    }

    #[test]
    fn epd_parses_semicolon_and_padding_variants() {
        // Stray `;` straight after the FEN, spaced check sign, NUL padding.
        let p = parse("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - ; bm Qd1 + ; id BK.01\0");
        assert_eq!(p.best_moves, vec!["Qd1"]);
        assert_eq!(p.id, "BK.01");

        // `;` glued to the EP field, annotated avoid move.
        let p = parse("2k5/2p5/1q1p4/pPpPp1p1/N1P1Pp1p/P4PbP/KQ4P1/8 w - -; am Nxb6 + ? ; c1 diff=10");
        assert_eq!(p.avoid_moves, vec!["Nxb6"]);
        assert!(p.best_moves.is_empty());

        // Counters followed by `;`, quoted id containing a space.
        let p = parse("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1 ; bm Qb3; id \"Kaufman 1\"");
        assert_eq!(p.best_moves, vec!["Qb3"]);
        assert_eq!(p.id, "Kaufman 1");

        let p = parse("startpos ; bm e4");
        assert!(p.fen.starts_with("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"));
        assert_eq!(p.best_moves, vec!["e4"]);
    }

    #[test]
    fn epd_c_file_pawn_moves_are_not_comment_opcodes() {
        // `c5` is a move here, not the `c5` comment opcode.
        let p = parse("r2r2k1/p1pnqpp1/1p2p2p/3b4/3P4/3BPN2/PP3PPP/2RQR1K1 b - - ; bm c5; id  sbd.095");
        assert_eq!(p.best_moves, vec!["c5"]);
        let p = parse("8/8/8/8/8/8/2P5/K6k w - - 0 1 bm c4 c3");
        assert_eq!(p.best_moves, vec!["c4", "c3"]);
        // ...while a real comment still opens its own operation.
        let p = parse("8/8/8/8/8/8/2P5/K6k w - - 0 1 bm c4 c0 \"note\"");
        assert_eq!(p.best_moves, vec!["c4"]);
    }

    #[test]
    fn epd_lines_without_solution_ops_are_not_gradable() {
        assert!(!parse("8/5kpp/8/8/1p3P2/6PP/r3KP2/1R1q4 w - -  0 1; D4 23441").is_gradable());
        assert!(!parse("5rkq/3r1p1p/5RpP/2p3P1/2B2Q2/p7/P4PK1/8 w - - 0 1; draw").is_gradable());
        assert!(parse("8/8/8/8/8/8/2P5/K6k w - - am c4").is_gradable());
    }

    #[test]
    fn epd_grading_rules() {
        let mut p = parse("8/8/8/8/8/8/2P5/K6k w - - 0 1 bm Rc8+ dm 3");
        // dm: any move that mates in time passes, even if it isn't the bm.
        assert!(grade(&p, "Qh7+", "g8h7", Some(3)).is_ok());
        assert!(grade(&p, "Rc8+", "c1c8", Some(4)).is_err());
        assert!(grade(&p, "Rc8+", "c1c8", None).is_err());
        // bm: SAN with check suffix, or UCI.
        p.direct_mate = None;
        assert!(grade(&p, "Rc8+", "c1c8", None).is_ok());
        assert!(grade(&p, "Rc7", "c1c7", None).is_err());
        // am applies alongside bm.
        p.avoid_moves = vec!["Rc8".to_string()];
        assert!(grade(&p, "Rc8+", "c1c8", None).is_err());
    }
}
