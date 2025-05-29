"""
Sample text for testing the vectorization agent.

This is a demonstration corpus containing various themes and topics
to test the semantic analysis capabilities of our AI-driven text
vectorization system.
"""

SAMPLE_TEXT = """
The Adventures of Tom Sawyer by Mark Twain

Tom Sawyer lived with his Aunt Polly and his half-brother Sid in the town of St. Petersburg, Missouri. Tom was a mischievous boy who often found himself in trouble, but his charm and wit usually helped him escape punishment.

One sunny morning, Tom was forced to whitewash a fence as punishment for his misdeeds. However, being the clever boy he was, Tom convinced his friends that whitewashing was actually a privilege and a joy. Soon, other boys were begging Tom to let them help, even offering their treasures in exchange for a chance to paint the fence.

The river that flowed past St. Petersburg was the mighty Mississippi, and it held endless fascination for the boys of the town. Tom and his best friend Huckleberry Finn would often dream of adventures on its waters, imagining themselves as pirates or explorers discovering new lands.

In the schoolhouse, Tom struggled with his lessons, finding arithmetic and grammar far less interesting than the adventures that awaited him outdoors. His teacher, Mr. Dobbins, was a stern man who believed that discipline and order were essential for learning.

One day, Tom witnessed a murder in the graveyard at midnight. The killer was Injun Joe, a dangerous criminal who had returned to seek revenge. Tom was terrified and swore Huck to secrecy, but the guilt of keeping silent weighed heavily on his conscience.

During the summer, Tom and Huck discovered a treasure map that led them on a thrilling adventure. They searched caves and hidden places, always one step behind the elusive treasure. Their friendship grew stronger through these shared experiences and dangers.

The town of St. Petersburg was a typical American frontier community, where everyone knew everyone else's business. The church played a central role in community life, and Sunday services were attended by all respectable families.

When Becky Thatcher arrived in town, Tom was immediately smitten. Her golden hair and bright blue eyes captured his heart, and he found himself showing off whenever she was near. Their young romance was filled with the innocent passions of childhood.

The story teaches us about the importance of friendship, courage, and moral growth. Tom's journey from a mischievous boy to a young man who stands up for justice reflects the timeless themes of coming of age and finding one's place in the world.
"""

def get_sample_text() -> str:
    """Return the sample text for testing."""
    return SAMPLE_TEXT

if __name__ == "__main__":
    # Save sample text to file
    import os
    os.makedirs("data", exist_ok=True)
    
    with open("data/sample_text.txt", "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
    
    print("Sample text saved to data/sample_text.txt")
    print(f"Text length: {len(SAMPLE_TEXT)} characters")
    print(f"Word count: {len(SAMPLE_TEXT.split())} words")
