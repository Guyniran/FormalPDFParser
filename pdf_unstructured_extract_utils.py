########## From UnstructuredIO #############
## It reads tables in sequence (but headers not concatenated). It also reads images.
# base is taken from: https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb

# from difflib import SequenceMatcher
from rapidfuzz import fuzz # type: ignore
from collections import Counter

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()
def similar(a, b):
    return fuzz.ratio(a, b) / 100 

######
# Remove first page and those that start with "Published Date:" or "Table of Contents" in the first text element at the first pages (the first can be Image, this we will skip)
# Remove the ones with "index" in the first text elements at the last pages.
######
def remove_first_pages(raw_pdf_elements):
    current_page = 1
    new_page_top_content = 0
    new_pdf_elements = []
    for i, element in enumerate(raw_pdf_elements):
        if element.metadata.page_number == 1:
            continue # Ignore cover page
        if element.category == 'PageBreak': 
            continue # Ignore leading page breaks
        
        # If a new page starts
        if element.metadata.page_number != current_page:
            current_page = element.metadata.page_number
            new_page_top_content = 1
            page_to_ignore = False

        # If currently processing the top content of a new page
        if new_page_top_content == 1:
            new_page_top_content = 0
            if element.text.lower().startswith('published date:') or element.text.lower().startswith("table of contents"):
                page_to_ignore = True
            elif element.category == 'Image' and (i + 1 < len(raw_pdf_elements)) \
                and (raw_pdf_elements[i + 1].text.lower().startswith('published date:') or raw_pdf_elements[i + 1].text.lower().startswith("table of contents")):
                page_to_ignore = True

        # If the page is not flagged to be ignored, add the element to the new list
        if not page_to_ignore:
            new_pdf_elements.append(element)

        # Stop further checks if a non-ignored page is found
        if not page_to_ignore and element.metadata.page_number == current_page:
            new_pdf_elements.extend([e for e in raw_pdf_elements[i+1:]])
            return new_pdf_elements

def remove_header_image(pdf_elements_wo_start):
    to_remove = set()
    number_of_total_elements = len(pdf_elements_wo_start)
    for i, element in enumerate(pdf_elements_wo_start):
        if i == 0:
            if element.category == "Image" and ("Allscripts" in element.text or "Altera" in element.text):
                to_remove.add(i)
            else:
                continue
        if element.category == 'PageBreak':
            # Ensure we don't go out of bounds
            if i + 1 < number_of_total_elements:
                next_element = pdf_elements_wo_start[i + 1]
                if next_element.category == "Image" and ("Allscripts" in next_element.text or "Altera" in next_element.text):
                    to_remove.add(i + 1)
                else:
                    continue

    return [elem for i, elem in enumerate(pdf_elements_wo_start) if i not in to_remove]

def remove_end_pages_from_index(elements):
    # Find page break positions
    page_break_positions = [i for i, x in enumerate(elements) if x.category == 'PageBreak']
    
    # Iterate through page breaks from the end to the beginning
    for j in range(len(page_break_positions) - 1, -1, -1):
        start = page_break_positions[j - 1] if j > 0 else 0
        # Check if there are elements after the page break
        if start + 1 < len(elements):
            # Check if the first element of the page contains "Index"
            if "Index" in elements[start + 1].text:
                # Remove all elements from the identified page break onwards
                elements = elements[:start + 1]
            else:
                break  # Stop processing further if "Index" is not found
    
    return elements

def find_minimum_footer_length(elements, page_break_positions, threshold, tolerance_percentage):
    """
    Identify the minimum footer length, allowing a percentage of outlier footers, and collect mismatched footers.

    Args:
        elements (list): A list of objects with a 'text' attribute.
        page_break_positions (list): Indices in 'elements' where pages break.
        threshold (float): Similarity threshold (0 to 1).
        tolerance_percentage (float): Allowed percentage of mismatched footers (0 to 100).

    Returns:
        (int, list): Minimum footer length or 0 if no consistent footer is found, 
                     and a list of mismatched footer texts.
    """
    total_pages = len(page_break_positions)
    max_mismatches = max(1, int((tolerance_percentage / 100) * total_pages))

    for num_elements in range(3, 0, -1):
        footer_texts = []

        # Collect footer texts for each page break
        for pos in page_break_positions:
            if pos - num_elements >= 0:
                footer_text = ''.join(
                    elements[pos - i].text for i in range(num_elements, 0, -1)
                )
                footer_texts.append(footer_text)

        # Count similar footers for each footer
        similarity_counts = Counter()
        mismatched_texts = []

        for i, footer_text in enumerate(footer_texts):
            similar_count = 0
            for other_footer_text in footer_texts:
                if similar(footer_text, other_footer_text) >= threshold:
                    similar_count += 1
            similarity_counts[footer_text] = similar_count

        # Determine the dominant group size
        max_similar_count = max(similarity_counts.values())
        
        # Collect mismatched footers
        for footer_text, count in similarity_counts.items():
            if count < max_similar_count:
                mismatched_texts.append(footer_text)

        # Check if mismatched count is within allowed tolerance
        if len(mismatched_texts) <= max_mismatches:
            return num_elements, mismatched_texts

    return 0, []

def remove_footer_elements(elements, threshold=0.8, tolerance_percentage = 7):
    page_break_positions = [i for i, x in enumerate(elements) if x.category == 'PageBreak']
    to_remove = set()
    
    min_footer_length, mismatched_texts  = find_minimum_footer_length(elements, page_break_positions, threshold, tolerance_percentage)
    # pirnt(mismatched_texts) # for debugging purpose
    if min_footer_length > 0:
        for pos in page_break_positions:
            if pos - min_footer_length >= 0:
                to_remove.update([pos-i for i in range(1, min_footer_length + 1)])

    return [elem for i, elem in enumerate(elements) if i not in to_remove]

def remove_pagebreak_elements(elements):
    page_break_positions = [i for i, x in enumerate(elements) if x.category == 'PageBreak']
    to_remove = set(page_break_positions)

    return [elem for i, elem in enumerate(elements) if i not in to_remove]

def remove_remaining_table_of_contents(elements):
    to_remove = set()
    for i, element in enumerate(elements):
        char_counts = Counter(element.text)
        most_common_char = ''
        if char_counts:
            most_common_char = char_counts.most_common(1)[0][0]
        ## From start - if this is a table that contains many dots, then it is a continuation of the Table of Contents
        if element.category == 'Table' and most_common_char == '.':
            to_remove.add(i)
        else:
            return [elem for i, elem in enumerate(elements) if i not in to_remove]
        