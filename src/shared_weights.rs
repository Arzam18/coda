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
/// Deterministic content hash used only to NAME the object, over a sparse
/// sample (one 64-byte line in every 64): hashing all ~92 MB would cost every
/// engine launch tens of milliseconds. Correctness never rests on the name —
/// attach compares every byte — so a sample collision only means falling back
/// to a private copy.
fn content_hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(bytes.len());
    for line in bytes.chunks(64).step_by(64) {
        h.write(line);
    }
    h.finish()
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PREFIX: &str = "coda-w1-";

/// The host's huge-page policy for shared memory (the bracketed value of
/// /sys/kernel/mm/transparent_hugepage/shmem_enabled). With "never" a shared
/// copy lives on 4 KB pages, where a private copy can have 2 MB ones.
pub fn shmem_thp_policy() -> String {
    std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/shmem_enabled")
        .ok()
        .and_then(|s| s.split('[').nth(1).and_then(|t| t.split(']').next()).map(str::to_string))
        .unwrap_or_else(|| "n/a".into())
}

/// Try to back `bytes` with a shared read-only mapping holding identical
/// content. `tag` separates the different arrays of one net in the name.
/// On failure returns the reason, so the caller can report why it fell back
/// to a private copy instead of failing silently.
pub fn share(bytes: &[u8], tag: &str) -> Result<SharedRegion, String> {
    if bytes.is_empty() {
        return Err("empty array".into());
    }
    if !enabled() {
        return Err("disabled".into());
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
        let path = std::ffi::CString::new(name).map_err(|_| "bad name".to_string())?;
        return match linux::attach(&path, bytes) {
            Ok(r) => Ok(r),
            // Not published yet: become the publisher.
            Err(e) if e.starts_with("open: No such file") => linux::publish_then_attach(&path, bytes),
            Err(e) => Err(e),
        };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = tag;
        Err("not supported on this platform".into())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{SharedRegion, PREFIX};
    use std::ffi::{CStr, CString};

    const MADV_COLLAPSE: libc::c_int = 25;
    const HUGE: usize = 2 * 1024 * 1024;

    /// Map `len` bytes of `fd` read-only at a 2 MB-aligned address, so the
    /// kernel can back the mapping with huge pages.
    fn map_aligned(fd: libc::c_int, len: usize) -> Result<*mut libc::c_void, String> {
        unsafe {
            // Reserve address space with room to align, then map over it.
            let span = len + HUGE;
            let r = libc::mmap(std::ptr::null_mut(), span, libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0);
            if r == libc::MAP_FAILED {
                return Err(errno("mmap reserve"));
            }
            let base = r as usize;
            let aligned = (base + HUGE - 1) & !(HUGE - 1);
            let p = libc::mmap(aligned as *mut libc::c_void, len, libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED, fd, 0);
            if p == libc::MAP_FAILED {
                let e = errno("mmap");
                libc::munmap(r, span);
                return Err(e);
            }
            // Release the unused head and tail of the reservation.
            if aligned > base {
                libc::munmap(r, aligned - base);
            }
            let end = (aligned + len + 4095) & !4095;
            if base + span > end {
                libc::munmap(end as *mut libc::c_void, base + span - end);
            }
            Ok(p)
        }
    }

    fn errno(what: &str) -> String {
        format!("{}: {}", what, std::io::Error::last_os_error())
    }

    fn stat_ino(fd: libc::c_int) -> Option<(u64, u64, i64)> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            return None;
        }
        Some((st.st_ino as u64, st.st_nlink as u64, st.st_size as i64))
    }

    /// Map `fd` read-only and accept it only if its content equals `bytes`.
    fn map_verified(fd: libc::c_int, path: &CStr, bytes: &[u8]) -> Result<SharedRegion, String> {
        let (_, nlink, size) = stat_ino(fd).ok_or_else(|| errno("fstat"))?;
        if nlink == 0 {
            return Err("object was removed".into());
        }
        if size != bytes.len() as i64 {
            return Err(format!("size mismatch ({} vs {})", size, bytes.len()));
        }
        let p = map_aligned(fd, bytes.len())?;
        let mapped = unsafe { std::slice::from_raw_parts(p as *const u8, bytes.len()) };
        if mapped != bytes {
            unsafe { libc::munmap(p, bytes.len()) };
            return Err("content mismatch".into());
        }
        // Ask for 2 MB pages. MADV_COLLAPSE (Linux 6.1+) builds them in the
        // shared page cache regardless of the host's shmem THP policy, and
        // every process whose mapping is 2 MB-aligned then maps them huge.
        // The first caller pays the collapse; later ones find it done.
        // Without huge pages the shared copy runs on 4 KB pages, which costs
        // measurable speed against a private THP copy.
        unsafe {
            libc::madvise(p, bytes.len(), libc::MADV_HUGEPAGE);
            // A collapse racing another process's attach can fail with
            // EAGAIN; a couple of retries settle it.
            for _ in 0..3 {
                if libc::madvise(p, bytes.len(), MADV_COLLAPSE) == 0 {
                    break;
                }
                if std::io::Error::last_os_error().raw_os_error() != Some(libc::EAGAIN) {
                    break;
                }
            }
        }
        Ok(SharedRegion { ptr: p as *mut u8, len: bytes.len(), fd, path: path.to_owned() })
    }

    /// Attach to an already-published object.
    pub fn attach(path: &CStr, bytes: &[u8]) -> Result<SharedRegion, String> {
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(errno("open"));
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
        let result = if locked {
            map_verified(fd, path, bytes)
        } else {
            Err("lock busy".into())
        };
        if result.is_err() {
            unsafe { libc::close(fd) };
        }
        result
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
    pub fn publish_then_attach(path: &CStr, bytes: &[u8]) -> Result<SharedRegion, String> {
        collect_garbage(path);
        let tmp_name = format!(
            "{}.tmp{}",
            path.to_str().map_err(|_| "bad name".to_string())?,
            std::process::id()
        );
        let tmp = CString::new(tmp_name).map_err(|_| "bad name".to_string())?;
        let fd = unsafe {
            libc::open(
                tmp.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(errno("create"));
        }
        // Hold the shared lock from creation on: the garbage collector of a
        // concurrent publisher then cannot remove this file mid-publish.
        let written = if unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) } != 0 {
            Err(errno("flock"))
        } else {
            write_all(fd, bytes)
        };
        if let Err(e) = written {
            unsafe {
                libc::unlink(tmp.as_ptr());
                libc::close(fd);
            }
            return Err(e);
        }
        let linked = unsafe { libc::link(tmp.as_ptr(), path.as_ptr()) } == 0;
        unsafe { libc::unlink(tmp.as_ptr()) };
        if linked {
            return match map_verified(fd, path, bytes) {
                Ok(r) => Ok(r),
                Err(e) => {
                    // Could not map what we just wrote: withdraw it.
                    unsafe { libc::unlink(path.as_ptr()) };
                    unsafe { libc::close(fd) };
                    Err(e)
                }
            };
        }
        unsafe { libc::close(fd) };
        attach(path, bytes)
    }

    fn write_all(fd: libc::c_int, bytes: &[u8]) -> Result<(), String> {
        if unsafe { libc::ftruncate(fd, bytes.len() as libc::off_t) } != 0 {
            return Err(errno("ftruncate"));
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
                // ENOSPC here usually means /dev/shm is smaller than the net
                // (e.g. a container's default 64 MB).
                return Err(errno("write"));
            }
            off += n as usize;
        }
        Ok(())
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
        assert!(share(&[], &tag).is_err());
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
