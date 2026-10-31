//! Local client access to files the daemon creates.
//!
//! Clients do not stream data through the daemon: they open the socket, then read and write
//! arena files and RAM-tier chunk files directly at the paths the daemon reports. When the
//! daemon runs in a container (usually as root) and clients run on the host or in other pods
//! (usually as an unprivileged user), those files must be shared with the clients' group.
//! `client_gid` names that group; without it, files keep the daemon's default ownership and
//! only the daemon's own user can use the cache.

use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use tracing::warn;

/// Mode for sockets, arenas and chunk files when a client group is configured.
pub const CLIENT_FILE_MODE: u32 = 0o660;
/// Mode for directories clients create files in (setgid: new files inherit the group).
pub const CLIENT_DIR_MODE: u32 = 0o2770;

fn apply(path: &Path, client_gid: Option<u32>, mode: u32) -> io::Result<()> {
    let Some(gid) = client_gid else {
        return Ok(());
    };
    std::os::unix::fs::chown(path, None, Some(gid))?;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// Gives `client_gid` read/write access to a socket or data file.
pub fn grant_file(path: &Path, client_gid: Option<u32>) -> io::Result<()> {
    apply(path, client_gid, CLIENT_FILE_MODE)
}

/// Gives `client_gid` access to a directory of chunk files.
pub fn grant_dir(path: &Path, client_gid: Option<u32>) -> io::Result<()> {
    apply(path, client_gid, CLIENT_DIR_MODE)
}

/// Like `grant_file`, but logs instead of failing: a cache the clients cannot open is still
/// usable by the daemon's own user, and the warning names the fix.
pub fn grant_file_or_warn(path: &Path, client_gid: Option<u32>) {
    if let Err(e) = grant_file(path, client_gid) {
        warn!(
            "Could not give group {:?} access to {:?}: {}. Clients running as another user will get \
             permission errors (the daemon must run as root or be a member of client_gid).",
            client_gid, path, e
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn my_gid(dir: &Path) -> u32 {
        let probe = dir.join("gid_probe");
        fs::write(&probe, b"").unwrap();
        fs::metadata(&probe).unwrap().gid()
    }

    #[test]
    fn test_grant_sets_group_and_modes() {
        let dir = std::env::temp_dir().join(format!("hc_access_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let gid = my_gid(&dir);
        let file = dir.join("arena.dat");
        fs::write(&file, b"x").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();

        grant_file(&file, Some(gid)).unwrap();
        let meta = fs::metadata(&file).unwrap();
        assert_eq!(meta.gid(), gid);
        assert_eq!(meta.mode() & 0o7777, CLIENT_FILE_MODE);

        let sub = dir.join("shm");
        fs::create_dir_all(&sub).unwrap();
        grant_dir(&sub, Some(gid)).unwrap();
        assert_eq!(fs::metadata(&sub).unwrap().mode() & 0o7777, CLIENT_DIR_MODE);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_no_client_gid_leaves_modes_alone() {
        let dir = std::env::temp_dir().join(format!("hc_access_none_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("arena.dat");
        fs::write(&file, b"x").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o640)).unwrap();
        grant_file(&file, None).unwrap();
        assert_eq!(fs::metadata(&file).unwrap().mode() & 0o7777, 0o640);
        let _ = fs::remove_dir_all(&dir);
    }
}
