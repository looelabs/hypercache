//! Turns LMCache's native key strings into HyperCache URIs.
//!
//! The URI format must not change, or stored blocks can't be found. Every free-text
//! field is percent-encoded so none can add a path level, and salted and unsalted keys have
//! different shapes, so no salt can collide with "no salt".

/// Percent-encodes like Python's `urllib.parse.quote(value, safe="")`.
pub fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(byte as char),
            _ => {
                out.push('%');
                out.push(char::from_digit((byte >> 4) as u32, 16).unwrap().to_ascii_uppercase());
                out.push(char::from_digit((byte & 15) as u32, 16).unwrap().to_ascii_uppercase());
            }
        }
    }
    out
}

/// Maps `<model>@<rank hex>@<group hex>@<hash hex>[@<salt>]` to a URI. `quoted_namespace` must
/// already be quoted. Anything that doesn't parse is an error, which callers treat as a miss.
pub fn key_to_uri(quoted_namespace: &str, key: &str) -> Result<String, String> {
    let parts: Vec<&str> = key.split('@').collect();
    if parts.len() != 4 && parts.len() != 5 {
        return Err(format!("malformed key: {key}"));
    }
    let (model, rank, group, hash) = (parts[0], parts[1], parts[2], parts[3]);
    let rank = normalize_hex(rank).ok_or_else(|| format!("bad rank in key: {key}"))?;
    let group = normalize_hex(group).ok_or_else(|| format!("bad group in key: {key}"))?;
    if hash.is_empty() || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("bad hash in key: {key}"));
    }
    let salt = match parts.get(4) {
        None => "_".to_string(),
        Some(&"") => return Err(format!("empty salt in key: {key}")),
        Some(salt) => format!("s-{}", quote(salt)),
    };
    Ok(format!(
        "hypercache://kv_cache/{quoted_namespace}/{}/{rank}/{group}/{salt}/{hash}.kv",
        quote(model)
    ))
}

/// Drops zero padding: URIs use `{:x}`, while LMCache pads the rank to 8 digits.
fn normalize_hex(value: &str) -> Option<String> {
    let parsed = i128::from_str_radix(value, 16).ok()?;
    Some(if parsed < 0 {
        format!("-{:x}", -parsed)
    } else {
        format!("{parsed:x}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_quote() {
        // Expected values come from Python's urllib.parse.quote(value, safe="")
        assert_eq!(quote("Qwen/test-model"), "Qwen%2Ftest-model");
        assert_eq!(quote("a b+c~d_e.f"), "a%20b%2Bc~d_e.f");
        assert_eq!(quote("modèle"), "mod%C3%A8le");
    }

    #[test]
    fn unsalted_key_uri_format() {
        let uri = key_to_uri("ns", "Qwen/m@00000000@0@abc123").unwrap();
        assert_eq!(uri, "hypercache://kv_cache/ns/Qwen%2Fm/0/0/_/abc123.kv");
    }

    #[test]
    fn rank_and_group_lose_padding() {
        let uri = key_to_uri("ns", "m@0000001f@a@ff").unwrap();
        assert_eq!(uri, "hypercache://kv_cache/ns/m/1f/a/_/ff.kv");
    }

    #[test]
    fn negative_group_uri_format() {
        // Python formats -1 as "-1" with {:x}, and the rank's {:08x} as "-0000001"
        assert_eq!(
            key_to_uri("ns", "m@-0000001@-1@ff").unwrap(),
            "hypercache://kv_cache/ns/m/-1/-1/_/ff.kv"
        );
    }

    #[test]
    fn salt_cannot_collide_with_no_salt() {
        let salted = key_to_uri("ns", "m@0@0@ff@_").unwrap();
        assert_eq!(salted, "hypercache://kv_cache/ns/m/0/0/s-_/ff.kv");
        assert_ne!(salted, key_to_uri("ns", "m@0@0@ff").unwrap());
    }

    #[test]
    fn fields_cannot_add_path_levels() {
        let uri = key_to_uri("ns", "../../x@0@0@ff@a/b").unwrap();
        assert_eq!(uri, "hypercache://kv_cache/ns/..%2F..%2Fx/0/0/s-a%2Fb/ff.kv");
    }

    #[test]
    fn malformed_keys_are_errors() {
        for key in [
            "",
            "m@0@0",
            "m@0@0@ff@s@extra",
            "m@zz@0@ff",
            "m@0@zz@ff",
            "m@0@0@not-hex",
            "m@0@0@",
            "m@0@0@ff@",
        ] {
            assert!(key_to_uri("ns", key).is_err(), "{key} should be rejected");
        }
    }
}
