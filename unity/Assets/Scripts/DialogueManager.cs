using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;

namespace EmotionAwareNPCs
{
    /// <summary>
    /// Manages NPC dialogue based on emotion input
    /// </summary>
    public class DialogueManager : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private TextMeshProUGUI dialogueText;
        [SerializeField] private GameObject dialoguePanel;
        [SerializeField] private Image npcPortrait;
        
        [Header("Animation Settings")]
        [SerializeField] private float fadeInDuration = 0.5f;
        [SerializeField] private float fadeOutDuration = 0.3f;
        [SerializeField] private float displayDuration = 3.0f;
        
        [Header("Audio (Optional)")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip[] dialogueSounds;
        
        // Dialogue mapping
        private Dictionary<string, DialogueEntry> dialogueMap;
        
        // Animation state
        private bool isDisplayingDialogue = false;
        private Coroutine currentDialogueCoroutine;
        
        private void Awake()
        {
            InitializeDialogueMap();
            SetupUI();
        }
        
        private void OnEnable()
        {
            EmotionReader.OnDialogueTriggered += HandleDialogueTriggered;
        }
        
        private void OnDisable()
        {
            EmotionReader.OnDialogueTriggered -= HandleDialogueTriggered;
        }
        
        /// <summary>
        /// Initialize the emotion-to-dialogue mapping
        /// </summary>
        private void InitializeDialogueMap()
        {
            dialogueMap = new Dictionary<string, DialogueEntry>
            {
                {
                    "happy", new DialogueEntry
                    {
                        text = "Nice job! You're doing great!",
                        color = Color.green,
                        animationTrigger = "Happy"
                    }
                },
                {
                    "sad", new DialogueEntry
                    {
                        text = "Don't give up! You've got this!",
                        color = Color.blue,
                        animationTrigger = "Sad"
                    }
                },
                {
                    "angry", new DialogueEntry
                    {
                        text = "Try slowing down. Take a deep breath.",
                        color = Color.red,
                        animationTrigger = "Angry"
                    }
                },
                {
                    "fear", new DialogueEntry
                    {
                        text = "It's okay, take your time. I'm here to help.",
                        color = Color.yellow,
                        animationTrigger = "Fear"
                    }
                },
                {
                    "surprise", new DialogueEntry
                    {
                        text = "Wow, unexpected! That's interesting!",
                        color = Color.magenta,
                        animationTrigger = "Surprise"
                    }
                },
                {
                    "disgust", new DialogueEntry
                    {
                        text = "Let's try something else. How about this?",
                        color = Color.cyan,
                        animationTrigger = "Disgust"
                    }
                },
                {
                    "neutral", new DialogueEntry
                    {
                        text = "Keep going. You're on the right track.",
                        color = Color.white,
                        animationTrigger = "Neutral"
                    }
                }
            };
        }
        
        /// <summary>
        /// Setup UI components
        /// </summary>
        private void SetupUI()
        {
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }
            
            if (dialogueText != null)
            {
                dialogueText.text = "";
            }
        }
        
        /// <summary>
        /// Handle dialogue trigger from emotion reader
        /// </summary>
        private void HandleDialogueTriggered(string emotion)
        {
            if (dialogueMap.ContainsKey(emotion))
            {
                ShowDialogue(emotion);
            }
            else
            {
                Debug.LogWarning($"No dialogue found for emotion: {emotion}");
            }
        }
        
        /// <summary>
        /// Show dialogue for the given emotion
        /// </summary>
        public void ShowDialogue(string emotion)
        {
            if (isDisplayingDialogue)
            {
                StopCoroutine(currentDialogueCoroutine);
            }
            
            currentDialogueCoroutine = StartCoroutine(DisplayDialogueCoroutine(emotion));
        }
        
        /// <summary>
        /// Coroutine to display dialogue with animation
        /// </summary>
        private IEnumerator DisplayDialogueCoroutine(string emotion)
        {
            isDisplayingDialogue = true;
            
            DialogueEntry entry = dialogueMap[emotion];
            
            // Show dialogue panel
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(true);
            }
            
            // Set text and color
            if (dialogueText != null)
            {
                dialogueText.text = entry.text;
                dialogueText.color = entry.color;
            }
            
            // Play audio if available
            PlayDialogueAudio(emotion);
            
            // Trigger NPC animation
            TriggerNPCAnimation(entry.animationTrigger);
            
            // Fade in
            yield return StartCoroutine(FadeInDialogue());
            
            // Display for specified duration
            yield return new WaitForSeconds(displayDuration);
            
            // Fade out
            yield return StartCoroutine(FadeOutDialogue());
            
            // Hide dialogue panel
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }
            
            isDisplayingDialogue = false;
        }
        
        /// <summary>
        /// Fade in dialogue with animation
        /// </summary>
        private IEnumerator FadeInDialogue()
        {
            if (dialoguePanel == null) yield break;
            
            CanvasGroup canvasGroup = dialoguePanel.GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = dialoguePanel.AddComponent<CanvasGroup>();
            }
            
            float elapsedTime = 0f;
            while (elapsedTime < fadeInDuration)
            {
                elapsedTime += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(0f, 1f, elapsedTime / fadeInDuration);
                yield return null;
            }
            
            canvasGroup.alpha = 1f;
        }
        
        /// <summary>
        /// Fade out dialogue with animation
        /// </summary>
        private IEnumerator FadeOutDialogue()
        {
            if (dialoguePanel == null) yield break;
            
            CanvasGroup canvasGroup = dialoguePanel.GetComponent<CanvasGroup>();
            if (canvasGroup == null) yield break;
            
            float elapsedTime = 0f;
            while (elapsedTime < fadeOutDuration)
            {
                elapsedTime += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(1f, 0f, elapsedTime / fadeOutDuration);
                yield return null;
            }
            
            canvasGroup.alpha = 0f;
        }
        
        /// <summary>
        /// Play dialogue audio if available
        /// </summary>
        private void PlayDialogueAudio(string emotion)
        {
            if (audioSource != null && dialogueSounds != null && dialogueSounds.Length > 0)
            {
                // Simple audio selection based on emotion
                int audioIndex = Mathf.Abs(emotion.GetHashCode()) % dialogueSounds.Length;
                audioSource.PlayOneShot(dialogueSounds[audioIndex]);
            }
        }
        
        /// <summary>
        /// Trigger NPC animation
        /// </summary>
        private void TriggerNPCAnimation(string animationTrigger)
        {
            // Find NPC animator in the scene
            Animator npcAnimator = FindObjectOfType<Animator>();
            if (npcAnimator != null)
            {
                npcAnimator.SetTrigger(animationTrigger);
            }
        }
        
        /// <summary>
        /// Test dialogue display (for debugging)
        /// </summary>
        [ContextMenu("Test Happy Dialogue")]
        public void TestHappyDialogue()
        {
            ShowDialogue("happy");
        }
        
        [ContextMenu("Test Sad Dialogue")]
        public void TestSadDialogue()
        {
            ShowDialogue("sad");
        }
        
        [ContextMenu("Test Angry Dialogue")]
        public void TestAngryDialogue()
        {
            ShowDialogue("angry");
        }
        
        /// <summary>
        /// Get dialogue text for emotion (for external use)
        /// </summary>
        public string GetDialogueText(string emotion)
        {
            if (dialogueMap.ContainsKey(emotion))
            {
                return dialogueMap[emotion].text;
            }
            return "Keep going!";
        }
        
        /// <summary>
        /// Check if dialogue is currently being displayed
        /// </summary>
        public bool IsDisplayingDialogue => isDisplayingDialogue;
    }
    
    /// <summary>
    /// Dialogue entry data structure
    /// </summary>
    [System.Serializable]
    public class DialogueEntry
    {
        public string text;
        public Color color;
        public string animationTrigger;
    }
}
