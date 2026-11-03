"""Small labelled set shared by the fit and recalibrate tests (written for these tests)."""

TOPICS = {
    "weather": ["will it rain tomorrow", "how hot is it outside", "is it going to snow this weekend",
                "do I need an umbrella today", "what's the forecast for Friday", "how windy is it now",
                "will it be sunny on Sunday", "is there a storm coming", "temperature tonight please",
                "is it cold in Madrid", "chance of rain this afternoon", "forecast for the next three days"],
    "music": ["play some jazz", "put on my workout playlist", "next song please", "play the new album by Rosalía",
              "turn the music up", "shuffle my favourites", "play something relaxing", "skip this track",
              "play classical music for studying", "put on the radio", "play songs from the eighties",
              "repeat this song"],
    "alarm": ["wake me up at seven", "set an alarm for 6:30", "cancel my morning alarm", "snooze for ten minutes",
              "alarm at noon please", "change my alarm to eight", "remind me with an alarm at 5", "turn off all alarms",
              "set a weekday alarm at 7:15", "what alarms do I have", "delete the 9 o'clock alarm",
              "alarm in two hours"],
    "lights": ["turn off the kitchen lights", "dim the living room", "switch on the bedroom lamp",
               "make the lights warmer", "lights off everywhere", "brighten the hallway", "turn the porch light on",
               "set the lights to 30 percent", "switch off the desk lamp", "turn on all the lights",
               "lower the lights in the bedroom", "lights on in the bathroom"],
}
TEXTS = [t for v in TOPICS.values() for t in v] + ["what's the capital of France", "tell me a joke",
                                                   "how many calories in an apple", "who won the match"]
LABELS = [k for k, v in TOPICS.items() for _ in v] + [None] * 4
