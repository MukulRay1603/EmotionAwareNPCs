# Unity Client - Emotion-Aware NPCs

## Overview
Unity game client that displays NPC responses based on player emotions.

## Directory Structure
```
unity/
├── Assets/
│   ├── Scripts/           # C# game logic
│   ├── Scenes/            # Unity scenes
│   ├── Prefabs/           # NPC prefabs
│   └── UI/                # Dialogue UI
├── ProjectSettings/       # Unity project configuration
└── README.md             # This file
```

## Components (To be implemented by P2)

### 1. Minimal Scene
- One NPC character
- Subtitle/dialogue box
- Camera setup

### 2. Emotion Reader
- Coroutine reading emotion.json (1 Hz)
- Alternative: HTTP client for API polling
- Parse JSON and extract emotion

### 3. Dialogue Mapping
```csharp
Dictionary<string, string> dialogueMap = new Dictionary<string, string>
{
    {"happy", "Nice job!"},
    {"sad", "Don't give up!"},
    {"angry", "Try slowing down."},
    {"fear", "It's okay, take your time."},
    {"surprise", "Wow, unexpected!"},
    {"disgust", "Let's try something else."},
    {"neutral", "Keep going."}
};
```

### 4. Cooldown System
- 2-second cooldown between responses
- Prevents dialogue spam
- Debouncing: 6-10 frame agreement

## Setup Instructions

### Unity Version
- Unity 2022.3 LTS or higher

### Installation
1. Open Unity Hub
2. Open project from `unity/` folder
3. Import required packages:
   - TextMeshPro (for UI)
   - Newtonsoft.Json (for JSON parsing)

### Running the Scene
1. Open `Scenes/MainScene.unity`
2. Press Play
3. Ensure backend is running at `localhost:8000`

## API Integration Options

### Option A: Read JSON File
```csharp
string json = File.ReadAllText("../cv/output/emotion.json");
EmotionData data = JsonUtility.FromJson<EmotionData>(json);
```

### Option B: HTTP API Client
```csharp
using UnityEngine.Networking;

IEnumerator GetEmotion()
{
    UnityWebRequest request = UnityWebRequest.Get("http://localhost:8000/infer");
    yield return request.SendWebRequest();
    
    if (request.result == UnityWebRequest.Result.Success)
    {
        EmotionData data = JsonUtility.FromJson<EmotionData>(request.downloadHandler.text);
        UpdateDialogue(data.emotion);
    }
}
```

## Expected Behavior
1. Poll emotion API/file every 1 second
2. Check cooldown timer
3. If cooldown expired and emotion changed:
   - Look up dialogue line
   - Display in subtitle box
   - Start 2-second cooldown

## Demo Requirements
- 10-second video showing:
  - NPC character visible
  - Subtitle text updating
  - At least 2-3 emotion changes
  - Proper cooldown behavior

## TODO (P2 Tasks)
- [ ] Create Unity scene with NPC
- [ ] Implement emotion reader coroutine
- [ ] Create dialogue mapping system
- [ ] Add cooldown timer
- [ ] Test with mock emotions
- [ ] Record demo video

## Performance Notes
- Update frequency: 1 Hz (every second)
- Cooldown: 2 seconds
- Target frame rate: 60 FPS
- UI should not block main thread

