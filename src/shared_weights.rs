//! Cross-process sharing of the large read-only NNUE weight arrays.
//!
//! Every Coda process builds the same weight matrices at load time (~66 MB of
//! threat rows, ~25 MB of PSQ rows). Several processes on one machine — the
//! concurrent games of a match, an OpenBench worker's dev and base copies —
//! would otherwise each hold a private copy, so the same hot rows compete for
//! the shared L3 once per process. Measured on a 16-core Zen 5 host: with k
//! engines pinned to one 32 MB L3 slice, DRAM demand fills per node double
//! with every halving of the per-engine share (1.4 at k=1 to 10.0 at k=8),
//! because one search touches ~40 MB of distinct weight rows. With a single
//! physical copy those rows occupy L3 once.
//!
//! Scheme (Linux only; everything else keeps the private copy):
//! - Each process still loads the net privately, exactly as before, so the
//!   bytes it expects are known locally.
//! - The shared object is a file in `/dev/shm` named by a hash of those final
//!   bytes, the element size and the length. A process maps it read-only and
//!   compares it byte for byte with its private copy before switching over, so
//!   a stale, partial or colliding object can never be used: on any mismatch
//!   or error the private copy is kept.
//! - Publishing writes a temporary file, then `link()`s it to the final name,
//!   which fails if the name already exists, so a finished object is never
//!   replaced under a reader.
//! - Every user holds a shared `flock` on the object for as long as it maps
//!   it. A process that can take the lock exclusively (non-blocking) knows it
//!   is the last user and unlinks the name. Locks die with their process, so
//!   a crash leaves at most an unlocked file, which a later publisher removes.
//!
//! No atomics or in-memory flags cross the process boundary: publication is a
//! filesystem `link()` after the file is fully written, and readers verify the
//! whole content, so there is no memory-ordering contract to get wrong on
//! weakly-ordered CPUs.

use std::sync::atomic::{AtomicBool, Ordering};

/// UCI `SharedWeights`. Consulted when a net is loaded.
pub static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire) && std::env::var("CODA_NO_SHARED_WEIGHTS").is_err()
}

/// A read-only shared mapping of one weight array. Dropping it unmaps the
/// memory and, if this process was the last user, removes the name.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct SharedRegion {
    ptr: *mut u8,
    len: usize,
    #[cfg(target_os = "linux")]
    fd: libc::c_int,
    #[cfg(target_os = "linux")]
    path: std::ffi::CString,
}

unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
impl SharedRegion {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
/// Deterministic content hash used only to NAME the object. Correctness never
/// rests on it: attach compares the full bytes.
fn content_hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PREFIX: &str = "coda-w1-";

/// Try to back `bytes` with a shared read-only mapping holding identical
/// content. `tag` separates the different arrays of one net in the name.
/// Returns None whenever sharing is unavailable or anything fails.
pub fn share(bytes: &[u8], tag: &str) -> Option<SharedRegion> {
    if bytes.is_empty() || !enabled() {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let name = format!(
            "/dev/shm/{}{}-{:016x}-{}",
            PREFIX,
            tag,
            content_hash(bytes),
            bytes.len()
        );
        let path = std::ffi::CString::new(name).ok()?;
        if let Some(r) = linux::attach(&path, bytes) {
            return Some(r);
        }
        return linux::publish_then_attach(&path, bytes);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = tag;
        None
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{SharedRegion, PREFIX};
    use std::ffi::{CStr, CString};

    fn stat_ino(fd: libc::c_int) -> Option<(u64, u64, i64)> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            return None;
        }
        Some((st.st_ino as u64, st.st_nlink as u64, st.st_size as i64))
    }

    /// Map `fd` read-only and accept it only if its content equals `bytes`.
    fn map_verified(fd: libc::c_int, path: &CStr, bytes: &[u8]) -> Option<SharedRegion> {
        let (_, nlink, size) = stat_ino(fd)?;
        if nlink == 0 || size != bytes.len() as i64 {
            return None;
        }
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes.len(),
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return None;
        }
        let mapped = unsafe { std::slice::from_raw_parts(p as *const u8, bytes.len()) };
        if mapped != bytes {
            unsafe { libc::munmap(p, bytes.len()) };
            return None;
        }
        // Huge pages when the host's shmem policy allows them; a no-op otherwise.
        unsafe { libc::madvise(p, bytes.len(), libc::MADV_HUGEPAGE) };
        Some(SharedRegion { ptr: p as *mut u8, len: bytes.len(), fd, path: path.to_owned() })
    }

    /// Attach to an already-published object.
    pub fn attach(path: &CStr, bytes: &[u8]) -> Option<SharedRegion> {
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        // Only a last-user cleanup ever holds the lock exclusively, and it does
        // so without blocking, so a short retry covers that window.
        let mut locked = false;
        for _ in 0..50 {
            if unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) } == 0 {
                locked = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if locked {
            if let Some(r) = map_verified(fd, path, bytes) {
                return Some(r);
            }
        }
        unsafe { libc::close(fd) };
        None
    }

    /// Remove unlocked leftovers of earlier runs (other nets, crashed
    /// publishers). Any file this process can lock exclusively has no users.
    fn collect_garbage(keep: &CStr) {
        let Ok(dir) = std::fs::read_dir("/dev/shm") else { return };
        for entry in dir.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.starts_with(PREFIX) {
                continue;
            }
            let full = format!("/dev/shm/{}", name);
            if full.as_bytes() == keep.to_bytes() {
                continue;
            }
            let Ok(c) = CString::new(full) else { continue };
            let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
            if fd < 0 {
                continue;
            }
            if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                unsafe { libc::unlink(c.as_ptr()) };
            }
            unsafe { libc::close(fd) };
        }
    }

    /// Write a new object and publish it under `path`; on losing a publish
    /// race, attach to the winner's object instead.
    pub fn publish_then_attach(path: &CStr, bytes: &[u8]) -> Option<SharedRegion> {
        collect_garbage(path);
        let tmp_name = format!(
            "{}.tmp{}",
            path.to_str().ok()?,
            std::process::id()
        );
        let tmp = CString::new(tmp_name).ok()?;
        let fd = unsafe {
            libc::open(
                tmp.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return None;
        }
        // Hold the shared lock from creation on: the garbage collector of a
        // concurrent publisher then cannot remove this file mid-publish.
        let ok = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) } == 0
            && write_all(fd, bytes);
        if !ok {
            unsafe {
                libc::unlink(tmp.as_ptr());
                libc::close(fd);
            }
            return None;
        }
        let linked = unsafe { libc::link(tmp.as_ptr(), path.as_ptr()) } == 0;
        unsafe { libc::unlink(tmp.as_ptr()) };
        if linked {
            if let Some(r) = map_verified(fd, path, bytes) {
                return Some(r);
            }
            // Could not map what we just wrote: withdraw it.
            unsafe { libc::unlink(path.as_ptr()) };
            unsafe { libc::close(fd) };
            return None;
        }
        unsafe { libc::close(fd) };
        attach(path, bytes)
    }

    fn write_all(fd: libc::c_int, bytes: &[u8]) -> bool {
        if unsafe { libc::ftruncate(fd, bytes.len() as libc::off_t) } != 0 {
            return false;
        }
        let mut off = 0usize;
        while off < bytes.len() {
            let n = unsafe {
                libc::pwrite(
                    fd,
                    bytes[off..].as_ptr() as *const libc::c_void,
                    bytes.len() - off,
                    off as libc::off_t,
                )
            };
            if n <= 0 {
                return false;
            }
            off += n as usize;
        }
        true
    }

    impl Drop for SharedRegion {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
                // Last user: the exclusive lock succeeds only when no other
                // process holds the object. Unlink only if the name still
                // refers to our inode, never a successor published under it.
                if libc::flock(self.fd, libc::LOCK_EX | libc::LOCK_NB) == 0 {
                    let mut st: libc::stat = std::mem::zeroed();
                    if let Some((ino, _, _)) = stat_ino(self.fd) {
                        if libc::stat(self.path.as_ptr(), &mut st) == 0 && st.st_ino as u64 == ino {
                            libc::unlink(self.path.as_ptr());
                        }
                    }
                }
                libc::close(self.fd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn share_attach_verify_and_cleanup() {
        let data: Vec<u8> = (0..300_000u32).map(|i| ((i % 251) ^ (i / 997)) as u8).collect();
        let tag = format!("test{}", std::process::id());
        let a = share(&data, &tag).expect("publish");
        let b = share(&data, &tag).expect("attach");
        assert_eq!(unsafe { std::slice::from_raw_parts(a.as_ptr(), data.len()) }, &data[..]);
        assert_eq!(unsafe { std::slice::from_raw_parts(b.as_ptr(), data.len()) }, &data[..]);
        // Different content must never attach to this object.
        let mut other = data.clone();
        other[12345] ^= 1;
        let c = share(&other, &tag).expect("distinct object");
        assert_ne!(a.as_ptr(), c.as_ptr());
        let path = format!("/dev/shm/{}{}-{:016x}-{}", PREFIX, tag, content_hash(&data), data.len());
        drop(a);
        assert!(std::path::Path::new(&path).exists(), "still in use by b");
        drop(b);
        assert!(!std::path::Path::new(&path).exists(), "last user unlinks");
        drop(c);
    }
}
