import hashlib


def get_short_hash(input_str):
    # Create a hashlib object using the SHA256 algorithm
    hash_object = hashlib.sha256()

    # Convert the input string to bytes
    input_bytes = input_str.encode("utf-8")

    # Update the hash object with the input bytes
    hash_object.update(input_bytes)

    # Retrieve the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    # Get the first 10 characters of the hash
    short_hash = hash_hex[:10]

    return short_hash
