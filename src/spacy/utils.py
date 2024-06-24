
def find_substring_positions(main_string, substring):
    start_index = main_string.find(substring)
    if start_index == -1:
        return None
    end_index = start_index + len(substring) - 1
    return (start_index, end_index)

def find_substring_positions_mutil(main_string, *substrings):
    positions = {}

    for substring in substrings:
        start_index = main_string.find(substring)
        if start_index != -1:
            end_index = start_index + len(substring) - 1
            positions[substring] = (start_index, end_index)

    return positions

if __name__ == '__main__':
    main_string = "Google's CEO Sundar Pichai introduced the new Pixel at Google I/O."
    result = find_substring_positions_mutil(main_string,"Google","Sundar Pichai")
    print(result)
