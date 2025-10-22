using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System;

namespace EmotionAwareNPCs
{
    /// <summary>
    /// Reads emotion data from the backend API and triggers NPC responses
    /// </summary>
    public class EmotionReader : MonoBehaviour
    {
        [Header("API Configuration")]
        [SerializeField] private string apiBaseUrl = "http://localhost:8000";
        [SerializeField] private float pollInterval = 1.0f; // Poll every 1 second
        
        [Header("Dialogue Settings")]
        [SerializeField] private float cooldownDuration = 2.0f; // 2 second cooldown
        [SerializeField] private int stabilityFrames = 6; // Require 6 consecutive frames to agree
        
        [Header("Debug")]
        [SerializeField] private bool enableDebugLogs = true;
        
        // Private fields
        private EmotionData currentEmotion;
        private EmotionData lastTriggeredEmotion;
        private float lastTriggerTime;
        private int stableFrameCount;
        private string lastStableEmotion;
        
        // Events
        public static event Action<EmotionData> OnEmotionChanged;
        public static event Action<string> OnDialogueTriggered;
        
        // Properties
        public EmotionData CurrentEmotion => currentEmotion;
        public bool IsInCooldown => Time.time - lastTriggerTime < cooldownDuration;
        
        private void Start()
        {
            StartCoroutine(PollEmotionAPI());
        }
        
        /// <summary>
        /// Continuously polls the emotion API
        /// </summary>
        private IEnumerator PollEmotionAPI()
        {
            while (true)
            {
                yield return StartCoroutine(GetEmotionFromAPI());
                yield return new WaitForSeconds(pollInterval);
            }
        }
        
        /// <summary>
        /// Makes HTTP request to get latest emotion
        /// </summary>
        private IEnumerator GetEmotionFromAPI()
        {
            string url = $"{apiBaseUrl}/infer";
            
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();
                
                if (request.result == UnityWebRequest.Result.Success)
                {
                    try
                    {
                        string jsonResponse = request.downloadHandler.text;
                        EmotionData emotionData = JsonUtility.FromJson<EmotionData>(jsonResponse);
                        
                        ProcessEmotionData(emotionData);
                        
                        if (enableDebugLogs)
                        {
                            Debug.Log($"Received emotion: {emotionData.emotion} (confidence: {emotionData.confidence:F2})");
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Error parsing emotion data: {e.Message}");
                    }
                }
                else
                {
                    Debug.LogError($"API request failed: {request.error}");
                }
            }
        }
        
        /// <summary>
        /// Processes the received emotion data and triggers dialogue if appropriate
        /// </summary>
        private void ProcessEmotionData(EmotionData emotionData)
        {
            currentEmotion = emotionData;
            OnEmotionChanged?.Invoke(emotionData);
            
            // Check for emotion stability
            if (emotionData.emotion == lastStableEmotion)
            {
                stableFrameCount++;
            }
            else
            {
                stableFrameCount = 1;
                lastStableEmotion = emotionData.emotion;
            }
            
            // Trigger dialogue if conditions are met
            if (ShouldTriggerDialogue(emotionData))
            {
                TriggerDialogue(emotionData);
            }
        }
        
        /// <summary>
        /// Determines if dialogue should be triggered based on stability and cooldown
        /// </summary>
        private bool ShouldTriggerDialogue(EmotionData emotionData)
        {
            // Check cooldown
            if (IsInCooldown)
            {
                return false;
            }
            
            // Check stability (require N consecutive frames to agree)
            if (stableFrameCount < stabilityFrames)
            {
                return false;
            }
            
            // Check if emotion has changed from last triggered
            if (lastTriggeredEmotion != null && lastTriggeredEmotion.emotion == emotionData.emotion)
            {
                return false;
            }
            
            // Check confidence threshold
            if (emotionData.confidence < 0.7f)
            {
                return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Triggers dialogue for the given emotion
        /// </summary>
        private void TriggerDialogue(EmotionData emotionData)
        {
            lastTriggeredEmotion = emotionData;
            lastTriggerTime = Time.time;
            stableFrameCount = 0; // Reset stability counter
            
            OnDialogueTriggered?.Invoke(emotionData.emotion);
            
            if (enableDebugLogs)
            {
                Debug.Log($"Dialogue triggered for emotion: {emotionData.emotion}");
            }
        }
        
        /// <summary>
        /// Manually trigger dialogue (for testing)
        /// </summary>
        [ContextMenu("Test Dialogue Trigger")]
        public void TestTriggerDialogue()
        {
            if (currentEmotion != null)
            {
                TriggerDialogue(currentEmotion);
            }
        }
        
        /// <summary>
        /// Get remaining cooldown time
        /// </summary>
        public float GetRemainingCooldown()
        {
            return Mathf.Max(0, cooldownDuration - (Time.time - lastTriggerTime));
        }
    }
    
    /// <summary>
    /// Data structure for emotion information from API
    /// </summary>
    [System.Serializable]
    public class EmotionData
    {
        public string emotion;
        public float confidence;
        public float timestamp;
        public EmotionFeatures features;
    }
    
    /// <summary>
    /// Additional features extracted from emotion analysis
    /// </summary>
    [System.Serializable]
    public class EmotionFeatures
    {
        public float valence;
        public float arousal;
        public float stress_level;
        public float fatigue_level;
        public HeadPose head_pose;
        public float eye_aspect_ratio;
        public float motion_intensity;
    }
    
    /// <summary>
    /// Head pose information
    /// </summary>
    [System.Serializable]
    public class HeadPose
    {
        public float yaw;
        public float pitch;
        public float roll;
    }
}
