import json
import argparse
from collections import defaultdict

# Eight high-level domains and their identifying keywords (lowercase)
DOMAIN_KEYWORDS = {
    "cooking": [
        "cook", "egg", "salad", "noodles", "pasta", "coffee", "milk", "tomato", "tea", "dinner", "plate",
        # Kitchen utensils
        "bowl", "spoon", "knife", "fork", "spatula", "whisk", "ladle", "chopstick", "tong",
        "pot", "pan", "kettle", "skillet", "wok", "cup", "mug", 
        "chopping board", "chopping_board", "cutting board", "grater", "peeler", "strainer", "colander", "sieve",
        "measuring spoon", "measuring cup", "scale",
        "mortar", "pestle", "scissors", "scissor", "dipper", "scoop", "skimmer",
        "rolling pin",
        # Ingredients and seasonings
        "onion", "garlic", "ginger", "celery", "cucumber", "carrot", "pepper", "chili", "basil",
        "salt", "sugar", "honey", "butter", "cheese", "curd", "cream",
        "oil", "vinegar", "sauce", "ketchup", "mustard", "syrup", "spice", "oregano", "cinnamon",
        "flour", "nut", "almond", "peanut", "sesame", "groundnut",
        "beef", "meat", "cherry", "lemon", "olive", "vegetable", "parsley", "cilantro", "scallion", "ciliary",
        "coriander", "molasses", "beer",
        "fish", "sriracha", "paprika", "turmeric", "curry",
        "omelet", "omelette", "fry", "fried", "scrambled", "boil", "stir",
        # Kitchen supplies and equipment
        "napkin", "towel", "tissue", "paper towel", "cloth", "rag",
        "lighter", "matches", "dispenser", "holder", "organizer", "bin", "waste", "trash", "thrash",
        "table", "stool", "tray", "storage", "container", "bottle", "pack", "carton",
        "dining", "kitchen", "recipe", "instruction",
        "chafing dish", "espresso machine", "timer", "knob", "control",
        "bucket", "trolley", "jug", "jar", "can", "drain", "wrap",
        "pakkad", "soap", "stainer", "steel thong", "matchbox", "beaker", "blue basket", "stainless steel bowel",
        "cell phone", "clipboard", "wooden clip board"
    ],
    "health": [
        "covid", "test", "swab", "cassette", "antigen", "rapid test", "nasal",
        "extraction buffer", "solution tube", "sterile", "cpr", "manikin", "dummy",
        "medical", "health", "clinic", "patient", "toothbrush", "quality card"
    ],
    "bike repair": [
        "wheel", "chain", "bike", "bicycle", "tire", "tyre", "tube", "pump",
        "brake", "pedal", "lever", "wrench", "spanner", "clamp",
        "handlebar", "fork", "seat", "stay", "cable", "sprocket", "caliper",
        "valve", "rim", "spoke"
    ],
    "music": [
        "violin", "piano", "guitar", "music"
    ],
    "basketball": [
        "basketball", "hoop"
    ],
    "soccer": [
        "soccer"
    ],
}

def categorize_object(obj_name: str) -> str:
    """Return one of the eight domain names based on keyword match; 'other' if none.

    Matching is case-insensitive and checks if any keyword is a substring of the
    provided object name. The mapping is heuristic but sufficient for aggregation.
    """
    name_l = str(obj_name).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name_l:
                return domain
    return "other"

def extract_and_categorize_objects(gt_file):
    """Extract all object names from a GT file and categorize them."""
    
    # Load GT file
    with open(gt_file, 'r') as fp:
        gt = json.load(fp)
    
    # Collect unique object IDs and their associated take IDs
    object_to_takes = defaultdict(set)
    
    # Iterate over all takes
    for take_id in gt["annotations"]:
        # Iterate over all objects in a take
        for object_id in gt["annotations"][take_id]["masks"].keys():
            object_to_takes[object_id].add(take_id)
    
    # Categorize
    categorized_objects = defaultdict(list)
    
    for obj in sorted(object_to_takes.keys()):
        category = categorize_object(obj)
        categorized_objects[category].append(obj)
    
    return categorized_objects, len(object_to_takes), object_to_takes

def print_results(categorized_objects, total_count, object_to_takes):
    """Print categorized results."""
    
    print("=" * 80)
    print(f"Found {total_count} unique objects in the GT file")
    print("=" * 80)
    print()
    
    # Print objects per category
    for category in list(DOMAIN_KEYWORDS.keys()) + ["other"]:
        if category in categorized_objects and len(categorized_objects[category]) > 0:
            print(f"[{category.upper()}] ({len(categorized_objects[category])} objects)")
            print("-" * 80)
            for obj in categorized_objects[category]:
                # For selected categories, also show take IDs
                if category == "music":
                    takes = sorted(list(object_to_takes[obj]))
                    print(f"  - {obj}")
                    print(f"    takes: {', '.join(takes)}")
                else:
                    print(f"  - {obj}")
            print()
    
    # Print summary stats
    print("=" * 80)
    print("Category stats:")
    print("=" * 80)
    for category in list(DOMAIN_KEYWORDS.keys()) + ["other"]:
        count = len(categorized_objects.get(category, []))
        if count > 0:
            percentage = (count / total_count) * 100
            print(f"  {category:15s}: {count:4d} ({percentage:5.1f}%)")
    print()

def save_to_json(categorized_objects, output_file):
    """Save results to a JSON file."""
    
    result = {}
    for category, objects in categorized_objects.items():
        result[category] = sorted(objects)
    
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    
    print(f"Saved results to: {output_file}")

def main(args):
    
    # Extract and categorize objects
    categorized_objects, total_count, object_to_takes = extract_and_categorize_objects(args.gt_file)
    
    # Print results
    print_results(categorized_objects, total_count, object_to_takes)
    
    # Save results if an output path is provided
    if args.output_file:
        save_to_json(categorized_objects, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract all objects from a GT file and categorize them')
    parser.add_argument('--gt-file', type=str, required=True, 
                        help="Path to the GT annotation file")
    parser.add_argument('--output-file', type=str, default=None,
                        help="Optional: output path to save the categorized results as JSON")
    args = parser.parse_args()
    
    main(args)

"""
python3 extract_gt_objects.py --gt-file /mnt/csp_sh/mmvision/home/shannanyan/egoexo4d_cor/correspondence/SegSwap/output/correspondence-gt.json --output-file ./correspondence-gt-objects.json
"""