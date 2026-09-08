//! Huge-page helpers shared by the TT and NNUE weight allocators.
//!
//! Two independent mechanisms can back a region with 2 MiB pages:
//! the explicit hugetlbfs pool (`MAP_HUGETLB`, `vm.nr_hugepages`) and
//! transparent huge pages (`MADV_HUGEPAGE` + fault-time or `MADV_COLLAPSE`
//! promotion). Third-party test hosts run any of THP `always` / `madvise` /
//! `never`, and only the pool works under `never`, so the allocators try the
//! pool first and fall back to THP. This module adds the two things that
//! design needs to behave well unattended:
//!
//! * `pool_can_fit`: only attempt the pool when the whole region fits in
//!   what is currently free, so a process never spends a failing syscall
//!   and, more importantly, the check is where a policy on sharing the pool
//!   between several engine processes belongs.
//! * `describe`: report what a region ACTUALLY ended up on, read back from
//!   `/proc/self/smaps`, so a tester's log shows `hugetlb`, `thp NN%` or
//!   `4k` for each region. Requesting huge pages and getting them are
//!   different things on an unknown host, and the difference is invisible
//!   from outside otherwise.

/// Bytes currently free in the explicit hugetlbfs pool (0 if none / non-Linux).
pub fn pool_free_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return 0 };
        let mut free_pages = 0usize;
        let mut page_kb = 2048usize;
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("HugePages_Free:") {
                free_pages = v.trim().parse().unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("Hugepagesize:") {
                page_kb = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(2048);
            }
        }
        free_pages * page_kb * 1024
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// True when the explicit pool can back a region of `bytes` in full.
pub fn pool_can_fit(bytes: usize) -> bool {
    pool_free_bytes() >= bytes
}

/// How a mapped region is backed, read from `/proc/self/smaps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// Explicit hugetlbfs pages (2 MiB kernel page size).
    Hugetlb,
    /// Transparent huge pages cover this percentage of the region.
    Thp(u8),
    /// Ordinary 4 KiB pages only.
    FourK,
    /// Could not be determined (non-Linux, unreadable smaps, empty region).
    Unknown,
}

pub fn backing(ptr: *const u8, len: usize) -> Backing {
    #[cfg(target_os = "linux")]
    {
        if len == 0 { return Backing::Unknown; }
        let Ok(s) = std::fs::read_to_string("/proc/self/smaps") else { return Backing::Unknown };
        let (lo, hi) = (ptr as usize, ptr as usize + len);
        let (mut in_range, mut overlap, mut anon_huge, mut huge_kernel) = (false, 0usize, 0usize, false);
        for line in s.lines() {
            let first = line.split(' ').next().unwrap_or("");
            if let Some((a, b)) = first.split_once('-') {
                if line.contains(' ') && !line.starts_with(|c: char| c.is_ascii_uppercase()) {
                    let vstart = usize::from_str_radix(a, 16).unwrap_or(0);
                    let vend = usize::from_str_radix(b, 16).unwrap_or(0);
                    in_range = vstart < hi && vend > lo;
                    if in_range { overlap += vend.min(hi) - vstart.max(lo); }
                    continue;
                }
            }
            if !in_range { continue; }
            if let Some(v) = line.strip_prefix("AnonHugePages:") {
                anon_huge += v.trim().trim_end_matches("kB").trim().parse::<usize>().unwrap_or(0) * 1024;
            } else if let Some(v) = line.strip_prefix("KernelPageSize:") {
                let kb: usize = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(4);
                if kb > 4 { huge_kernel = true; }
            }
        }
        if overlap == 0 { return Backing::Unknown; }
        if huge_kernel { return Backing::Hugetlb; }
        let pct = (anon_huge.min(len) * 100 / len) as u8;
        if pct == 0 { Backing::FourK } else { Backing::Thp(pct) }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        Backing::Unknown
    }
}

/// Short human form for `info string` lines.
pub fn describe(ptr: *const u8, len: usize) -> String {
    match backing(ptr, len) {
        Backing::Hugetlb => "hugetlb".to_string(),
        Backing::Thp(p) => format!("thp {}%", p),
        Backing::FourK => "4k".to_string(),
        Backing::Unknown => "unknown".to_string(),
    }
}
