//! Shared bearer-token authentication for the Controller and Node Agent HTTP APIs.

use axum::http::HeaderMap;

/// Constant-time byte comparison, so response timing does not reveal how much of a guessed
/// token matched.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Returns the configured token, treating blank values as "not configured".
pub fn effective_token(token: Option<&str>) -> Option<&str> {
    token.map(str::trim).filter(|t| !t.is_empty())
}

/// Validates `Authorization: Bearer <token>`. With no token configured every request passes;
/// callers that expose a network listener must refuse to run in that state.
pub fn validate_bearer(headers: &HeaderMap, expected_token: Option<&str>) -> bool {
    let Some(expected) = effective_token(expected_token) else {
        return true;
    };
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|token| constant_time_eq(token.trim().as_bytes(), expected.as_bytes()))
}

/// Attaches the bearer token to an outgoing request when one is configured.
pub fn with_bearer(req: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match effective_token(token) {
        Some(t) => req.bearer_auth(t),
        None => req,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bearer_validation() {
        let mut headers = HeaderMap::new();
        assert!(validate_bearer(&headers, None));
        assert!(validate_bearer(&headers, Some("  ")));
        assert!(!validate_bearer(&headers, Some("secret")));

        headers.insert("authorization", "Bearer wrong".parse().unwrap());
        assert!(!validate_bearer(&headers, Some("secret")));
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        assert!(validate_bearer(&headers, Some("secret")));
        assert!(validate_bearer(&headers, Some(" secret ")));
        headers.insert("authorization", "Basic secret".parse().unwrap());
        assert!(!validate_bearer(&headers, Some("secret")));

        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
