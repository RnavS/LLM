const MEDBRIEF_API_BASE = (() => {
  const explicit = String(
    window.MEDBRIEF_API_BASE ||
    localStorage.getItem('medbrief_api_base') ||
    ''
  ).trim().replace(/\/$/, '');
  if (explicit) return explicit;
  const localHostnames = new Set(['localhost', '127.0.0.1']);
  if (localHostnames.has(window.location.hostname)) {
    return window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';
  }
  return '';
})();

const RECALL_CONFIG = {
  API_BASE: MEDBRIEF_API_BASE,
  API_KEY: localStorage.getItem('recall_api_key') || 'local-model',
  API_ENDPOINT: `${MEDBRIEF_API_BASE}/api/chat/completions`,
  MODEL: 'medbrief-ai',
  MAX_TOKENS: 150,
  TEMPERATURE: 0.2,
  STREAM: true,

  MAX_CONTEXT_MESSAGES: 14,
  MEMORY_SUMMARY_THRESHOLD: 20,

  SYSTEM_PROMPT: '',

  CRISIS_KEYWORDS: [
    'kill myself', 'want to die', 'end my life', 'suicide', 'suicidal',
    'self-harm', 'self harm', 'cutting myself', 'hurt myself',
    'don\'t want to be alive', 'no reason to live', 'better off dead',
    'can\'t go on', 'end it all', 'take my life', 'not worth living',
    'going to hurt myself', 'plan to die', 'overdose', 'i should die',
    'i wanna die', 'feel like dying', 'suicide seems nice'
  ],

  CRISIS_RESPONSE_ADDENDUM: `

---
**If you're in crisis right now, please reach out:**
- 📞 **988 Suicide & Crisis Lifeline** — Call or text **988** (24/7)
- 💬 **Crisis Text Line** — Text **HOME** to **741741**
- 🚨 **Emergency** — Call **911**

You don't have to face this alone. A trained person can help right now.`,

  MOOD_LABELS: {
    5: 'great',
    4: 'okay',
    3: 'meh',
    2: 'low',
    1: 'struggling'
  }
};
