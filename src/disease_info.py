"""
LeafCheck - Disease Knowledge Base & Agricultural Recommendations
Provides domain-specific information, causes, symptoms, and treatments for 10 tomato conditions.
"""

from typing import Dict, Any

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Tomato___Bacterial_spot": {
        "disease_name": "Tomato Bacterial Spot",
        "causal_agent": "Xanthomonas perforans / Xanthomonas vesicatoria (Bacterium)",
        "symptoms": (
            "Small, circular or irregular dark brown to black spots on leaves, stems, and fruit. "
            "Lesions often have a faint yellow halo. Severe infection causes leaf defoliation and sunscald on fruit."
        ),
        "prevention": [
            "Use certified disease-free seeds and transplants.",
            "Practice crop rotation (avoid solanaceous crops for at least 2-3 years).",
            "Avoid overhead irrigation; use drip or furrow watering to keep foliage dry.",
            "Disinfect stakes, cages, and pruning equipment regularly."
        ],
        "treatment": [
            "Apply copper-based bactericides combined with mancozeb early in the season.",
            "Remove and destroy severely infected lower leaves.",
            "Apply biological controls containing Bacillus amyloliquefaciens."
        ]
    },
    "Tomato___Early_blight": {
        "disease_name": "Tomato Early Blight",
        "causal_agent": "Alternaria solani (Fungus)",
        "symptoms": (
            "Dark brown to black lesions showing distinctive concentric rings (target-board appearance). "
            "Typically begins on older, lower foliage and progresses upwards. Surrounding leaf tissue yellows."
        ),
        "prevention": [
            "Mulch heavily around the base of plants to prevent fungal spores from splashing from soil to leaves.",
            "Provide adequate plant spacing to maximize airflow.",
            "Stake and prune plants to keep foliage off the ground."
        ],
        "treatment": [
            "Apply protective fungicides like Chlorothalonil or Copper sulfate at first sign of infection.",
            "Use bio-fungicides based on Trichoderma harzianum.",
            "Prune infected bottom leaves immediately and discard away from compost."
        ]
    },
    "Tomato___Late_blight": {
        "disease_name": "Tomato Late Blight",
        "causal_agent": "Phytophthora infestans (Oomycete / Water Mold)",
        "symptoms": (
            "Large, irregularly shaped, water-soaked greenish-black lesions on leaves and stems. "
            "In humid conditions, a delicate white fungal growth appears on the undersides of leaves. Rapidly kills foliage."
        ),
        "prevention": [
            "Plant late-blight resistant cultivars if available in your region.",
            "Eliminate volunteer tomato and potato plants nearby.",
            "Monitor local agricultural blight forecasts (late blight thrives in cool, humid weather)."
        ],
        "treatment": [
            "Apply systemic fungicides (e.g., cymoxanil, mandipropamid) immediately upon detection.",
            "In organic cultivation, prophylactic copper sprays are essential before disease onset.",
            "Severely infested plants must be bagged and destroyed to prevent neighborhood epidemics."
        ]
    },
    "Tomato___Leaf_Mold": {
        "disease_name": "Tomato Leaf Mold",
        "causal_agent": "Passalora fulva / Fulvia fulva (Fungus)",
        "symptoms": (
            "Pale greenish-yellow chlorotic spots on the upper leaf surface without distinct margins. "
            "Undersides develop dense olive-green to grayish-brown velvety fungal mold."
        ),
        "prevention": [
            "Maintain relative humidity below 85% in greenhouses and tunnels.",
            "Increase plant spacing and improve air circulation with fans or row orientation.",
            "Avoid wetting foliage during irrigation."
        ],
        "treatment": [
            "Spray with sulfur, copper fungicides, or chlorothalonil.",
            "Use bio-fungicide sprays containing Bacillus subtilis.",
            "Ventilate greenhouse enclosures, especially during morning hours."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Septoria Leaf Spot",
        "causal_agent": "Septoria lycopersici (Fungus)",
        "symptoms": (
            "Numerous small, circular spots (1-3 mm) with dark brown margins and sunken, light tan or gray centers. "
            "Tiny black specks (pycnidia) are visible in the center of mature spots."
        ),
        "prevention": [
            "Rotate crops on a 2 to 3-year cycle.",
            "Eradicate nightshade weed species around tomato plots.",
            "Deep plow or destroy crop residues after harvest to bury overwintering spores."
        ],
        "treatment": [
            "Apply copper-based fungicides or mancozeb at 7-10 day intervals during wet periods.",
            "Prune off infected lower leaves early.",
            "Ensure strict drip irrigation practices."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease_name": "Two-Spotted Spider Mites Infestation",
        "causal_agent": "Tetranychus urticae (Arachnid Pest)",
        "symptoms": (
            "Fine pale stippling or yellow speckles across leaf surfaces. Leaves turn bronzed or scorched. "
            "Delicate silken webbing appears on the undersides of leaves and between stems under heavy infestation."
        ),
        "prevention": [
            "Maintain adequate soil moisture; mites thrive in hot, dry, and dusty conditions.",
            "Regularly inspect undersides of leaves using a 10x hand lens.",
            "Preserve natural predatory insects by avoiding broad-spectrum synthetic pyrethroids."
        ],
        "treatment": [
            "Apply neem oil, insecticidal soap, or horticultural oil spray thoroughly to leaf undersides.",
            "Introduce predatory mites (e.g., Phytoseiulus persimilis) in protected cultivation.",
            "Hose down plants with a strong stream of water to knock down mite colonies."
        ]
    },
    "Tomato___Target_Spot": {
        "disease_name": "Tomato Target Spot",
        "causal_agent": "Corynespora cassiicola (Fungus)",
        "symptoms": (
            "Small water-soaked pinpoint spots that expand into circular lesions with light brown centers "
            "and dark concentric rings, resembling a target. Causes extensive premature leaf drop."
        ),
        "prevention": [
            "Ensure proper nitrogen and potassium fertility management.",
            "Increase plant spacing for aeration.",
            "Rotate fields with non-host crops."
        ],
        "treatment": [
            "Apply broad-spectrum fungicides such as azoxystrobin, pyraclostrobin, or chlorothalonil.",
            "Remove diseased foliage to reduce canopy density.",
            "Maintain clean field boundaries free of alternative weed hosts."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "causal_agent": "Begomovirus transmitted by Silverleaf Whitefly (Bemisia tabaci)",
        "symptoms": (
            "Severe upward curling and cupping of leaflets, marked yellowing (chlorosis) along leaf margins, "
            "reduced leaf size, and severe plant stunting. Flower drop occurs and fruit set stops."
        ),
        "prevention": [
            "Use TYLCV-resistant or tolerant hybrid varieties.",
            "Install 50-mesh insect netting in greenhouses to exclude whiteflies.",
            "Use yellow sticky traps to detect and monitor whitefly populations early."
        ],
        "treatment": [
            "No chemical cure exists for viruses once a plant is infected.",
            "Control whitefly vectors using systemic insecticides (e.g., imidacloprid) or organic soaps.",
            "Immediately rogued and bag-infected plants to prevent vector transmission to healthy plants."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease_name": "Tomato Mosaic Virus (ToMV)",
        "causal_agent": "Tobamovirus (Mechanically transmitted virus)",
        "symptoms": (
            "Mottling of light and dark green patterns on leaves, leaf distortion, blistered lamina, "
            "and shoestringing/fern-like leaf narrowing. Fruit may ripen unevenly with internal browning."
        ),
        "prevention": [
            "Use certified virus-indexed seeds.",
            "Wash hands and tools with a 20% non-fat dry milk solution or 10% bleach before handling plants.",
            "Prohibit tobacco use near plants (tobacco mosaic virus can cross-contaminate)."
        ],
        "treatment": [
            "Viruses cannot be cured with chemical sprays.",
            "Immediately remove and destroy symptomatic plants.",
            "Clean and sanitize all stakes, trellis lines, and scissors before moving to healthy rows."
        ]
    },
    "Tomato___healthy": {
        "disease_name": "Healthy Tomato Leaf",
        "causal_agent": "None (Plant is healthy)",
        "symptoms": (
            "Uniform green leaf blade with normal turgor, no chlorotic speckles, necrotic spots, "
            "curling, or fungal/bacterial lesions."
        ),
        "prevention": [
            "Continue standard integrated pest management (IPM) practices.",
            "Maintain balanced watering schedule (avoid drought stress and waterlogging).",
            "Monitor regularly for early signs of disease or pest infestation."
        ],
        "treatment": [
            "No treatment needed. Maintain optimal soil nutrition and moisture."
        ]
    }
}


def get_disease_details(class_name: str) -> Dict[str, Any]:
    """Retrieve structured disease details for a given class label."""
    if class_name in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[class_name]
    
    # Fallback for unexpected or generic labels
    return {
        "disease_name": class_name.replace("___", " - ").replace("_", " "),
        "causal_agent": "Unknown",
        "symptoms": "No specific symptom data available for this class.",
        "prevention": ["Consult an agricultural extension specialist."],
        "treatment": ["Inspect plant foliage and isolate if symptomatic."]
    }
