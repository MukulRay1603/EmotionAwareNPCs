using System.Collections;
using UnityEngine;
using UnityEngine.AI;

namespace EmotionAwareNPCs
{
    /// <summary>
    /// Controls NPC behavior and animations based on emotion input
    /// </summary>
    public class NPCController : MonoBehaviour
    {
        [Header("NPC Components")]
        [SerializeField] private Animator animator;
        [SerializeField] private NavMeshAgent agent;
        [SerializeField] private Transform playerTransform;
        
        [Header("Movement Settings")]
        [SerializeField] private float followDistance = 3f;
        [SerializeField] private float moveSpeed = 2f;
        [SerializeField] private float rotationSpeed = 5f;
        
        [Header("Animation Settings")]
        [SerializeField] private float idleAnimationInterval = 5f;
        [SerializeField] private string[] idleAnimations = { "Idle1", "Idle2", "LookAround" };
        
        // Private fields
        private Vector3 lastPlayerPosition;
        private float lastIdleTime;
        private bool isFollowingPlayer = false;
        private Coroutine movementCoroutine;
        
        // Animation hashes for performance
        private int idleHash = Animator.StringToHash("Idle");
        private int walkHash = Animator.StringToHash("Walk");
        private int happyHash = Animator.StringToHash("Happy");
        private int sadHash = Animator.StringToHash("Sad");
        private int angryHash = Animator.StringToHash("Angry");
        private int fearHash = Animator.StringToHash("Fear");
        private int surpriseHash = Animator.StringToHash("Surprise");
        private int disgustHash = Animator.StringToHash("Disgust");
        private int neutralHash = Animator.StringToHash("Neutral");
        
        private void Start()
        {
            InitializeNPC();
            StartCoroutine(IdleBehavior());
        }
        
        private void OnEnable()
        {
            EmotionReader.OnEmotionChanged += HandleEmotionChanged;
            EmotionReader.OnDialogueTriggered += HandleDialogueTriggered;
        }
        
        private void OnDisable()
        {
            EmotionReader.OnEmotionChanged -= HandleEmotionChanged;
            EmotionReader.OnDialogueTriggered -= HandleDialogueTriggered;
        }
        
        /// <summary>
        /// Initialize NPC components and settings
        /// </summary>
        private void InitializeNPC()
        {
            // Get components if not assigned
            if (animator == null)
                animator = GetComponent<Animator>();
            
            if (agent == null)
                agent = GetComponent<NavMeshAgent>();
            
            // Find player if not assigned
            if (playerTransform == null)
            {
                GameObject player = GameObject.FindGameObjectWithTag("Player");
                if (player != null)
                    playerTransform = player.transform;
            }
            
            // Configure agent
            if (agent != null)
            {
                agent.speed = moveSpeed;
                agent.stoppingDistance = followDistance * 0.8f;
            }
            
            lastIdleTime = Time.time;
        }
        
        /// <summary>
        /// Handle emotion changes for subtle behavior adjustments
        /// </summary>
        private void HandleEmotionChanged(EmotionData emotionData)
        {
            // Adjust movement speed based on player's stress level
            if (agent != null && emotionData.features != null)
            {
                float stressLevel = emotionData.features.stress_level;
                agent.speed = Mathf.Lerp(moveSpeed * 0.5f, moveSpeed * 1.5f, stressLevel);
            }
            
            // Adjust follow distance based on player's emotional state
            if (emotionData.emotion == "fear" || emotionData.emotion == "sad")
            {
                // Get closer to comfort the player
                agent.stoppingDistance = followDistance * 0.5f;
            }
            else if (emotionData.emotion == "angry")
            {
                // Give more space
                agent.stoppingDistance = followDistance * 1.2f;
            }
            else
            {
                // Normal distance
                agent.stoppingDistance = followDistance * 0.8f;
            }
        }
        
        /// <summary>
        /// Handle dialogue triggers for major animations
        /// </summary>
        private void HandleDialogueTriggered(string emotion)
        {
            PlayEmotionAnimation(emotion);
            
            // Stop current movement to focus on dialogue
            if (agent != null)
            {
                agent.ResetPath();
            }
        }
        
        /// <summary>
        /// Play animation based on emotion
        /// </summary>
        private void PlayEmotionAnimation(string emotion)
        {
            if (animator == null) return;
            
            // Reset all emotion triggers first
            animator.ResetTrigger(happyHash);
            animator.ResetTrigger(sadHash);
            animator.ResetTrigger(angryHash);
            animator.ResetTrigger(fearHash);
            animator.ResetTrigger(surpriseHash);
            animator.ResetTrigger(disgustHash);
            animator.ResetTrigger(neutralHash);
            
            // Trigger appropriate animation
            switch (emotion.ToLower())
            {
                case "happy":
                    animator.SetTrigger(happyHash);
                    break;
                case "sad":
                    animator.SetTrigger(sadHash);
                    break;
                case "angry":
                    animator.SetTrigger(angryHash);
                    break;
                case "fear":
                    animator.SetTrigger(fearHash);
                    break;
                case "surprise":
                    animator.SetTrigger(surpriseHash);
                    break;
                case "disgust":
                    animator.SetTrigger(disgustHash);
                    break;
                case "neutral":
                default:
                    animator.SetTrigger(neutralHash);
                    break;
            }
        }
        
        /// <summary>
        /// Idle behavior coroutine
        /// </summary>
        private IEnumerator IdleBehavior()
        {
            while (true)
            {
                yield return new WaitForSeconds(idleAnimationInterval);
                
                // Play random idle animation if not in dialogue
                if (animator != null && !IsInDialogue())
                {
                    string randomIdle = idleAnimations[Random.Range(0, idleAnimations.Length)];
                    animator.SetTrigger(randomIdle);
                }
            }
        }
        
        /// <summary>
        /// Check if NPC is currently in dialogue
        /// </summary>
        private bool IsInDialogue()
        {
            DialogueManager dialogueManager = FindObjectOfType<DialogueManager>();
            return dialogueManager != null && dialogueManager.IsDisplayingDialogue;
        }
        
        /// <summary>
        /// Start following the player
        /// </summary>
        public void StartFollowingPlayer()
        {
            if (playerTransform == null || agent == null) return;
            
            isFollowingPlayer = true;
            if (movementCoroutine != null)
            {
                StopCoroutine(movementCoroutine);
            }
            movementCoroutine = StartCoroutine(FollowPlayerCoroutine());
        }
        
        /// <summary>
        /// Stop following the player
        /// </summary>
        public void StopFollowingPlayer()
        {
            isFollowingPlayer = false;
            if (movementCoroutine != null)
            {
                StopCoroutine(movementCoroutine);
                movementCoroutine = null;
            }
            
            if (agent != null)
            {
                agent.ResetPath();
            }
        }
        
        /// <summary>
        /// Follow player coroutine
        /// </summary>
        private IEnumerator FollowPlayerCoroutine()
        {
            while (isFollowingPlayer && playerTransform != null)
            {
                float distanceToPlayer = Vector3.Distance(transform.position, playerTransform.position);
                
                if (distanceToPlayer > followDistance)
                {
                    // Move towards player
                    if (agent != null)
                    {
                        agent.SetDestination(playerTransform.position);
                        animator.SetBool(walkHash, true);
                    }
                }
                else
                {
                    // Stop moving
                    if (agent != null)
                    {
                        agent.ResetPath();
                        animator.SetBool(walkHash, false);
                    }
                    
                    // Face the player
                    FacePlayer();
                }
                
                yield return new WaitForSeconds(0.1f);
            }
            
            // Stop walking animation
            if (animator != null)
            {
                animator.SetBool(walkHash, false);
            }
        }
        
        /// <summary>
        /// Make NPC face the player
        /// </summary>
        private void FacePlayer()
        {
            if (playerTransform == null) return;
            
            Vector3 direction = (playerTransform.position - transform.position).normalized;
            direction.y = 0; // Keep NPC upright
            
            if (direction != Vector3.zero)
            {
                Quaternion targetRotation = Quaternion.LookRotation(direction);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
        }
        
        /// <summary>
        /// Test emotion animation (for debugging)
        /// </summary>
        [ContextMenu("Test Happy Animation")]
        public void TestHappyAnimation()
        {
            PlayEmotionAnimation("happy");
        }
        
        [ContextMenu("Test Sad Animation")]
        public void TestSadAnimation()
        {
            PlayEmotionAnimation("sad");
        }
        
        [ContextMenu("Test Angry Animation")]
        public void TestAngryAnimation()
        {
            PlayEmotionAnimation("angry");
        }
        
        /// <summary>
        /// Get current follow state
        /// </summary>
        public bool IsFollowingPlayer => isFollowingPlayer;
    }
}
