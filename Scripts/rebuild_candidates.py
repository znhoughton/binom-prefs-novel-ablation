"""
rebuild_candidates.py
---------------------
Adds new candidate pairs, alphabetises word1/word2 within each pair
(word1 = alphabetically first), deduplicates, and writes back.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH  = ROOT / "Data" / "candidates.csv"
OUT_PATH = ROOT / "Data" / "candidates_new.csv"

# ── New pairs to add ─────────────────────────────────────────────────────────
# Listed as (word1, word2) — within-pair order will be fixed automatically.

NEW_PAIRS = [
    # Musical instruments: large/serious vs toy/small
    ("bassoons", "kazoos"),
    ("cellos", "kazoos"),
    ("cellos", "triangles"),
    ("clarinets", "kazoos"),
    ("flutes", "kazoos"),
    ("flutes", "tambourines"),
    ("harps", "kazoos"),
    ("harps", "tambourines"),
    ("organs", "kazoos"),
    ("organs", "whistles"),
    ("pianos", "kazoos"),
    ("pianos", "triangles"),
    ("saxophones", "kazoos"),
    ("saxophones", "whistles"),
    ("trombones", "whistles"),
    ("trumpets", "kazoos"),
    ("trumpets", "whistles"),
    ("tubas", "whistles"),
    ("violas", "kazoos"),
    ("violins", "kazoos"),
    ("violins", "tambourines"),
    ("oboes", "tambourines"),
    ("lutes", "kazoos"),
    ("harpsichords", "kazoos"),

    # Animals: more size contrasts
    ("gnats", "elephants"),
    ("gnats", "gorillas"),
    ("gnats", "rhinoceroses"),
    ("gnats", "hippopotamuses"),
    ("gnats", "mammoths"),
    ("krill", "whales"),
    ("mice", "tigers"),
    ("mice", "wolves"),
    ("minnows", "orcas"),
    ("rabbits", "wolves"),
    ("rabbits", "lions"),
    ("rabbits", "leopards"),
    ("rabbits", "jaguars"),
    ("bears", "rabbits"),
    ("cheetahs", "gazelles"),
    ("condors", "sparrows"),
    ("crocodiles", "tadpoles"),
    ("falcons", "wrens"),
    ("foxes", "rabbits"),
    ("gorillas", "sparrows"),
    ("leopards", "gazelles"),
    ("owls", "mice"),
    ("pelicans", "sardines"),
    ("seals", "herrings"),
    ("storks", "frogs"),
    ("vultures", "finches"),
    ("whales", "sardines"),
    ("cobras", "mice"),
    ("cougars", "deer"),
    ("wolves", "sparrows"),
    ("bison", "sparrows"),
    ("walruses", "crabs"),

    # Trees: tall trees vs small plants
    ("banyans", "ferns"),
    ("beeches", "acorns"),
    ("birches", "ferns"),
    ("cedars", "mosses"),
    ("chestnuts", "ferns"),
    ("cypresses", "mosses"),
    ("elms", "mushrooms"),
    ("larches", "ferns"),
    ("maples", "mushrooms"),
    ("oaks", "mushrooms"),
    ("pines", "mushrooms"),
    ("sequoias", "ferns"),
    ("spruces", "ferns"),
    ("walnuts", "ferns"),
    ("yews", "mosses"),

    # Luxury fabrics vs coarse/plain
    ("burlap", "cashmere"),
    ("burlap", "velvet"),
    ("canvas", "silk"),
    ("canvas", "chiffon"),
    ("cashmere", "flannel"),
    ("cashmere", "wool"),
    ("cotton", "taffeta"),
    ("damask", "linen"),
    ("lace", "twine"),
    ("linen", "satin"),
    ("muslin", "velvet"),
    ("burlap", "organza"),
    ("sackcloth", "velvet"),
    ("silk", "tweed"),
    ("brocade", "canvas"),

    # More gems vs common rock/dirt
    ("alexandrite", "gravel"),
    ("amber", "gravel"),
    ("aquamarine", "shale"),
    ("chalk", "tanzanite"),
    ("chrysoberyl", "gravel"),
    ("citrine", "gravel"),
    ("clay", "jade"),
    ("clay", "malachite"),
    ("coral", "gravel"),
    ("gravel", "moonstone"),
    ("gravel", "onyx"),
    ("gravel", "turquoise"),
    ("mud", "platinum"),
    ("peridot", "silt"),
    ("gravel", "spinel"),

    # Celestial bodies vs tiny lights
    ("candles", "galaxies"),
    ("comets", "sparks"),
    ("fireflies", "supernovas"),
    ("fireflies", "moons"),
    ("lanterns", "quasars"),
    ("meteors", "raindrops"),
    ("moths", "pulsars"),
    ("nebulae", "sparks"),
    ("planets", "sparks"),
    ("sparks", "stars"),
    ("asteroids", "pebbles"),
    ("fireflies", "galaxies"),
    ("candles", "nebulae"),

    # Weather: more power contrasts
    ("blizzards", "mists"),
    ("breezes", "monsoons"),
    ("breezes", "typhoons"),
    ("cyclones", "gusts"),
    ("drizzles", "hailstorms"),
    ("drizzles", "typhoons"),
    ("embers", "thunderbolts"),
    ("eruptions", "sparks"),
    ("gales", "sparks"),
    ("mudslides", "puddles"),
    ("puffs", "squalls"),
    ("showers", "typhoons"),
    ("ripples", "waterspouts"),
    ("drizzles", "thunderstorms"),
    ("whimpers", "earthquakes"),

    # Structures: grand vs humble
    ("abbeys", "hovels"),
    ("arenas", "sheds"),
    ("basilicas", "hovels"),
    ("castles", "hovels"),
    ("coliseums", "shacks"),
    ("hovels", "palaces"),
    ("mausoleums", "shacks"),
    ("hovels", "temples"),
    ("aqueducts", "ditches"),
    ("pyramids", "sandcastles"),
    ("birdhouses", "skyscrapers"),
    ("cathedrals", "outhouses"),
    ("closets", "dungeons"),
    ("birdhouses", "turrets"),
    ("ballrooms", "hovels"),

    # Social status: more high vs low
    ("admirals", "paupers"),
    ("archdukes", "vagrants"),
    ("barons", "paupers"),
    ("bishops", "vagabonds"),
    ("counts", "vagabonds"),
    ("dukes", "vagrants"),
    ("earls", "paupers"),
    ("generals", "conscripts"),
    ("kings", "mendicants"),
    ("knights", "serfs"),
    ("lords", "mendicants"),
    ("nobles", "paupers"),
    ("beggars", "priests"),
    ("paupers", "princes"),
    ("serfs", "viscounts"),
    ("sultans", "vagabonds"),
    ("tsars", "vagrants"),
    ("pharaohs", "scullions"),
    ("emperors", "scullions"),
    ("queens", "serfs"),

    # Weapons: more power contrasts
    ("battleaxes", "penknives"),
    ("cannons", "darts"),
    ("cannons", "slingshots"),
    ("claymores", "penknives"),
    ("flintlocks", "slingshots"),
    ("muskets", "slingshots"),
    ("pikes", "slingshots"),
    ("bombs", "pebbles"),
    ("catapults", "slingshots"),
    ("crossbows", "slingshots"),

    # Geographic scale contrasts
    ("chasms", "puddles"),
    ("cliffs", "pools"),
    ("gorges", "puddles"),
    ("deltas", "puddles"),
    ("oceans", "ponds"),
    ("clearings", "prairies"),
    ("meadows", "steppes"),
    ("glades", "taigas"),
    ("bogs", "tundras"),
    ("canyons", "gullies"),
    ("cliffs", "gullies"),
    ("lakes", "oceans"),
    ("continents", "islands"),
    ("hillocks", "mountains"),
    ("deserts", "oases"),

    # Luxury/prestige vs humble (food, royal items, misc)
    ("banquets", "crumbs"),
    ("crumbs", "feasts"),
    ("champagne", "vinegar"),
    ("champagne", "water"),
    ("nectar", "swill"),
    ("feasts", "scraps"),
    ("coronets", "rags"),
    ("diadems", "pebbles"),
    ("hairpins", "tiaras"),
    ("mantles", "rags"),
    ("stools", "thrones"),
    ("mansions", "pigsties"),
    ("ballrooms", "cellars"),
    ("feasts", "morsels"),
    ("goblets", "puddles"),
    ("palaces", "pigsties"),
    ("tapestries", "rags"),
    ("chandeliers", "candles"),
    ("chandeliers", "torches"),
    ("banners", "rags"),

    # User-supplied additions
    ("abashed", "sorry"),
    ("actresses", "lumberjacks"),
    ("allergic", "unaccustomed"),
    ("annoying", "teal"),
    ("bacteria", "candy"),
    ("beautiful", "stinky"),
    ("bicycles", "robots"),
    ("bishops", "seamstresses"),
    ("bitter", "purple"),
    ("blankets", "kittens"),
    ("campfires", "wildfires"),
    ("chanting", "enchanting"),
    ("chauffeurs", "stewardesses"),
    ("cherries", "llamas"),
    ("chickens", "fences"),
    ("coroners", "senators"),
    ("currant", "pomegranate"),
    ("deposed", "murdered"),
    ("determined", "forgettable"),
    ("discontent", "tearfulness"),
    ("disheveled", "dreary"),
    ("donates", "provides"),
    ("felines", "quails"),
    ("first", "ninety-eighth"),
    ("flowers", "zinnias"),
    ("fuming", "mad"),
    ("gelatin", "lard"),
    ("groundskeeper", "superintendent"),
    ("happily", "rudely"),
    ("hesitate", "readjust"),
    ("horses", "loons"),
    ("jacket", "phone"),
    ("kale", "vegetables"),
    ("lankier", "lanky"),
    ("litter", "newts"),
    ("masculine", "undignified"),
    ("marooned", "missing"),
    ("nurses", "patriarchs"),
    ("puppies", "tigers"),
    ("rats", "sharks"),
    ("therapy", "vacations"),
    ("vocabulary", "vowels"),
]


def normalise(w1, w2):
    """Return (alphabetically_first, alphabetically_second)."""
    a, b = w1.strip().lower(), w2.strip().lower()
    return (a, b) if a <= b else (b, a)


def main():
    # Read existing pairs
    existing = set()
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() in ("word1", ""):
                continue
            existing.add(normalise(row[0], row[1]))

    # Merge new pairs
    all_pairs = set(existing)
    for w1, w2 in NEW_PAIRS:
        all_pairs.add(normalise(w1, w2))

    added = len(all_pairs) - len(existing)
    print(f"Existing: {len(existing)}  |  New unique: {added}  |  Total: {len(all_pairs)}")

    # Write sorted by word1 then word2
    sorted_pairs = sorted(all_pairs)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2"])
        w.writerows(sorted_pairs)

    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
