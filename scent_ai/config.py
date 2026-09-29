CARTRIDGES = [
    "pine",
    "ocean", 
    "smoke",
    "earth",
    "rain",
    "asphalt",
]

SCENT_SOURCES = {
    "coniferous_forest": {
        "positive": [
            "a photo of evergreen pine trees with needle leaves",
            "a detailed view of a dense coniferous forest with spruce and pine bark",
            "green pine trees canopy in a forest under daylight",
            "woodland terrain with pine needles and evergreen conifers",
        ],
        "negative": [
            "a photo of dark gray paved asphalt road",
            "thick black smoke billowing from a burning fire",
            "a bright orange fireball explosion",
            "open blue ocean sea water with waves",
            "a city street with concrete buildings",
        ],
    },
    
    "broadleaf_forest": {
        "positive": [
            "a photo of lush green deciduous forest with broad leaves",
            "broadleaf woodland foliage with green oak and maple trees",
            "a detailed view of green tree leaves in a sunny forest",
            "dense jungle vegetation with green broadleaf trees",
        ],
        "negative": [
            "sharp evergreen pine tree needles",
            "dark gray paved asphalt street",
            "black smoke and bright red fire",
            "deep blue ocean water surface",
        ],
    },
    
    "ocean_sea": {
        "positive": [
            "a photo of open deep blue ocean water surface",
            "a sea coast with blue water waves and white foam",
            "a sandy beach shoreline next to ocean water",
            "coastal sea water view under open sky",
        ],
        "negative": [
            "dense green forest with pine trees",
            "dry brown dirt and muddy path",
            "dark asphalt city road",
            "thick black fire smoke",
        ],
    },
    
    "rain_wetness": {
        "positive": [
            "a photo of heavy rain streaks falling through the air",
            "a wet asphalt road covered with rain puddles and glossy reflections",
            "a stormy rainy day scene with water droplets",
            "wet ground surface during a heavy rainfall storm",
        ],
        "negative": [
            "dry asphalt road under clear sun",
            "bright clear sunny blue sky with no clouds",
            "dry dusty ground soil",
            "bright red flames and thick smoke",
        ],
    },
    
    "smoke_fire": {
        "positive": [
            "a photo of raging bright orange fire with black smoke plumes",
            "dense jet-black smoke billowing from an intense blaze",
            "active burning fire with glowing embers and dark smoke",
            "wildfire with visible red and yellow flames",
        ],
        "negative": [
            "a calm green pine forest landscape",
            "a regular gray vehicle or pickup truck parked in nature",
            "a peaceful man standing outdoors under green trees",
            "a clear blue sky with green tree canopy",
            "clean dry asphalt road with no fire",
        ],
    },
        
    "soil_mud": {
        "positive": [
            "a photo of bare brown dirt soil ground",
            "a muddy ground terrain with wet brown earth",
            "a bumpy dirt track with dark brown soil and mud",
            "wet muddy ground with exposed brown earth",
        ],
        "negative": [
            "dark gray paved asphalt pavement",
            "clean smooth concrete surface",
            "thick black fire smoke and flames",
            "green grass meadow with wild flowers",
        ],
    },
    
    "asphalt_city": {
        "positive": [
            "a photo of dark gray paved asphalt road surface",
            "an urban asphalt highway with painted white lane markings",
            "a city street with dark asphalt pavement and buildings",
            "an asphalt parking lot with black smooth pavement",
        ],
        "negative": [
            "a brown dirt forest path",
            "green grass field with pine trees",
            "raging fire flames and smoke",
            "wet brown mud soil",
        ],
    },
    
    "combat_action": {
        "positive": [
            "a photo of bright muzzle flashes from military rifles in combat",
            "active military gunfight battle scene with gun smoke",
            "tactical warfare action with gunfire sparks and shooting",
        ],
        "negative": [
            "a peaceful nature landscape with green trees",
            "a calm person walking in a quiet forest",
            "a vehicle parked near pine trees",
            "clear blue sky over countryside",
        ],
    },
    
    "racing_speed": {
        "positive": [
            "a photo of a race car drifting on asphalt track with white tire friction smoke",
            "high speed sports car driving fast on paved race track",
            "vehicle spinning tires creating white rubber smoke on asphalt",
        ],
        "negative": [
            "a person walking peacefully in a green forest",
            "a static truck parked in woods",
            "raging red flames from a burning fire",
        ],
    },
    
    "explosions": {
        "positive": [
            "a photo of a massive bright orange fireball explosion burst",
            "an explosive detonation with flying shockwave debris and fire",
            "a fiery bomb explosion with bright yellow and red blast flames",
        ],
        "negative": [
            "a peaceful green pine forest landscape",
            "a truck parked outdoors in the woods",
            "a man standing calmly near a car",
            "a clear blue sky with green trees",
        ],
    },
}

SOURCE_TO_CARTRIDGE = {
    "coniferous_forest": {
        "pine": 1.0,
        "earth": 0.25,
    },
    "broadleaf_forest": {
        "earth": 0.60,
    },
    "ocean_sea": {
        "ocean": 1.0,
    },
    "rain_wetness": {
        "rain": 1.0,
        "earth": 0.15,
        "asphalt": 0.15,
    },
    "smoke_fire": {
        "smoke": 1.0,
    },
    "soil_mud": {
        "earth": 1.0,
    },
    "asphalt_city": {
        "asphalt": 1.0,
    },
    "combat_action": {
        "smoke": 0.35,
        "asphalt": 0.15,
    },
    "racing_speed": {
        "asphalt": 0.85,
        "smoke": 0.15,
    },
    "explosions": {
        "smoke": 1.0,
        "earth": 0.20,
    },
}