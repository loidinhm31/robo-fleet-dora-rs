use robo_rover_lib::MAX_TTS_TEXT_CHARS;

pub fn sanitize_text(input: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut in_tag = false;
    for character in input.replace('\0', " ").chars() {
        match character {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => output.push(character),
            _ => {}
        }
    }
    let collapsed = output.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return Err("TTS text must not be empty".to_string());
    }
    if collapsed.chars().count() > MAX_TTS_TEXT_CHARS {
        return Err(format!("TTS text exceeds {MAX_TTS_TEXT_CHARS} characters"));
    }
    Ok(collapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_markup_and_collapses_whitespace() {
        assert_eq!(
            sanitize_text(" Hello <laugh>  world <bad").unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn rejects_empty_text_after_markup_removal() {
        assert!(sanitize_text(" <laugh> ").is_err());
    }
}
