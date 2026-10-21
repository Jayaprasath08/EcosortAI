"""Waste image classifier built on a pretrained MobileNetV2 (ImageNet).

MobileNetV2 is NOT trained on waste classes. It predicts ImageNet labels (e.g. "water_bottle"),
and this module maps those labels to waste categories by keyword and sums their probabilities.
That is a transfer shortcut that works for obvious items; for real accuracy fine-tune the model
on a waste dataset (TrashNet / TACO) - see README, "Improving accuracy".
"""
import numpy as np

CATEGORY_KEYWORDS = {
    "plastic": ["water_bottle", "pop_bottle", "plastic_bag", "pill_bottle", "lotion", "soap_dispenser",
                "water_jug", "bucket", "sunscreen", "shower_cap", "diaper", "packet", "hamper"],
    "paper":   ["carton", "envelope", "paper_towel", "toilet_tissue", "book_jacket", "notebook", "binder",
                "menu", "comic_book", "crossword", "jigsaw", "tissue"],
    "metal":   ["milk_can", "frying_pan", "wok", "caldron", "saucepan", "ladle", "can_opener", "strainer",
                "nail", "screw", "hammer", "safety_pin", "padlock", "chain", "tin", "Dutch_oven"],
    "glass":   ["beer_bottle", "wine_bottle", "red_wine", "vase", "goblet", "beer_glass", "perfume",
                "pitcher", "whiskey_jug", "cocktail_shaker"],
    "organic": ["banana", "orange", "lemon", "strawberry", "pineapple", "pomegranate", "fig", "jackfruit",
                "custard_apple", "Granny_Smith", "corn", "cucumber", "broccoli", "cauliflower", "zucchini",
                "squash", "artichoke", "bell_pepper", "cabbage", "mushroom", "head_cabbage"],
}
MIN_MAPPED_PROBABILITY = 0.05   # below this, the image is reported as "uncertain"


class ModelUnavailable(Exception):
    """Raised when TensorFlow or the pretrained weights cannot be loaded."""


def map_labels(labels_with_probs):
    """labels_with_probs: iterable of (imagenet_label, probability) -> result dict (pure function, testable)."""
    scores = {c: 0.0 for c in CATEGORY_KEYWORDS}
    best_label = {c: ("", 0.0) for c in CATEGORY_KEYWORDS}
    top_label = ("", 0.0)
    for label, p in labels_with_probs:
        if p > top_label[1]:
            top_label = (label, p)
        for cat, words in CATEGORY_KEYWORDS.items():
            if any(f"_{w.lower()}_" in f"_{label.lower()}_" for w in words):  # whole-word match
                scores[cat] += p
                if p > best_label[cat][1]:
                    best_label[cat] = (label, p)
    mapped_total = sum(scores.values())
    if mapped_total < MIN_MAPPED_PROBABILITY:
        return {"category": "other", "confidence": 0.0, "top_label": top_label[0],
                "note": "No waste-related ImageNet label scored high enough; please choose the category manually."}
    cat = max(scores, key=scores.get)
    return {"category": cat, "confidence": round(scores[cat] / mapped_total, 3),
            "top_label": best_label[cat][0],
            "note": "Estimated from ImageNet labels; confirm or correct before saving."}


class WasteClassifier:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from tensorflow.keras.applications import MobileNetV2
            self._model = MobileNetV2(weights="imagenet")
        except Exception as exc:  # no TensorFlow, or weights could not be downloaded
            raise ModelUnavailable(str(exc)) from exc

    def predict(self, image_path):
        self._load()
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        from PIL import Image
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        x = preprocess_input(np.expand_dims(np.asarray(img, dtype="float32"), 0))
        preds = self._model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=50)[0]
        return map_labels([(label, float(p)) for _, label, p in decoded])


classifier = WasteClassifier()
