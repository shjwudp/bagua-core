pub fn get_rank() -> i32 {
    return std::env::var("RANK")
        .unwrap_or("0".to_string())
        .parse::<i32>()
        .unwrap();
}
