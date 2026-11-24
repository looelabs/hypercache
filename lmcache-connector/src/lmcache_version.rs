//! Which LMCache releases this connector has been tested with.
//!
//! The connector plugs into LMCache's internal `native_plugin` interface, which changes between
//! releases (later versions make `get_memory_tags` abstract and change `ObjectKey`). An untested
//! version may still work, so the connector only warns; it never refuses to start.

/// LMCache releases the connector's safety and failure tests have passed on.
pub const TESTED_LMCACHE_VERSIONS: &[&str] = &["0.5.5", "0.5.6", "0.5.7.post1"];

/// The warning for an installed LMCache `version`, or None when that release was tested.
///
/// A local build tag (`0.5.7.post1+cu128`) counts as its release. Anything else, including dev,
/// rc and later post releases, is untested.
pub fn untested_version_problem(version: &str) -> Option<String> {
    let version = version.trim();
    let release = version.split('+').next().unwrap_or(version);
    if TESTED_LMCACHE_VERSIONS.contains(&release) {
        return None;
    }
    Some(format!(
        "LMCache {version} has not been tested with this connector (tested: {}). LMCache's plugin \
         interface changes between releases. Check that stores and loads hit and that answers \
         match before relying on it, or install a tested version",
        TESTED_LMCACHE_VERSIONS.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tested_versions_pass() {
        for v in TESTED_LMCACHE_VERSIONS {
            assert_eq!(untested_version_problem(v), None, "{v}");
        }
        assert_eq!(untested_version_problem(" 0.5.6\n"), None);
    }

    #[test]
    fn local_build_tag_counts_as_its_release() {
        assert_eq!(untested_version_problem("0.5.7.post1+cu128"), None);
        assert!(untested_version_problem("0.5.8+cu128").is_some());
    }

    #[test]
    fn untested_versions_warn() {
        for v in [
            "0.5.7",
            "0.5.7.post2",
            "0.5.8",
            "0.6.0",
            "0.5.7.dev12",
            "0.5.8rc1",
            "0.5.55",
            "1.0.0",
        ] {
            let problem = untested_version_problem(v).unwrap_or_else(|| panic!("{v} passed"));
            assert!(problem.contains(v), "{problem}");
            assert!(problem.contains("0.5.5, 0.5.6, 0.5.7.post1"), "{problem}");
        }
    }
}
