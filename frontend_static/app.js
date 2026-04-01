document.addEventListener('DOMContentLoaded', () => {
  const memory = new RecallMemory();
  const engine = new RecallEngine(memory);
  const API_BASE = RECALL_CONFIG.API_BASE || '';
  const apiUrl = path => `${API_BASE}${path}`;

  let currentConversationId = null;
  let currentTheme = localStorage.getItem('recall_theme') || 'dark';
  let runtimeConfig = null;
  let currentApiKeyId = localStorage.getItem('recall_api_key_id') || '';
  let authSession = { authenticated: false, user: null, mode: 'signed-out' };
  let authMode = 'login';

  const $ = id => document.getElementById(id);
  const welcomeScreen = $('welcome-screen');
  const chatMessages = $('chat-messages');
  const messagesContainer = $('messages-container');
  const messageInput = $('message-input');
  const sendBtn = $('send-btn');
  const newChatBtn = $('new-chat-btn');
  const sidebarToggle = $('sidebar-toggle');
  const sidebar = $('sidebar');
  const conversationList = $('conversation-list');
  const themeToggle = $('theme-toggle');
  const themeIconDark = $('theme-icon-dark');
  const themeIconLight = $('theme-icon-light');
  const moodCheckBtn = $('mood-check-btn');
  const moodModal = $('mood-modal');
  const moodClose = $('mood-close');
  const crisisBtn = $('crisis-btn');
  const crisisModal = $('crisis-modal');
  const crisisClose = $('crisis-close');
  const headerStatus = $('header-status');
  const welcomePrompts = $('welcome-prompts');
  const settingsBtn = $('settings-btn');
  const settingsModal = $('settings-modal');
  const settingsClose = $('settings-close');
  const developerSection = $('developer-section');
  const generateApiKeyBtn = $('generate-api-key');
  const revokeApiKeyBtn = $('revoke-api-key');
  const apiKeyDisplay = $('api-key-display');
  const apiKeyValue = $('api-key-value');
  const apiKeyCopy = $('api-key-copy');
  const clearMemoryBtn = $('clear-memory');
  const prefMaxTokens = $('pref-max-tokens');
  const prefTemperature = $('pref-temperature');
  const sessionEmail = $('session-email');
  const sessionMode = $('session-mode');
  const sessionPill = $('session-pill');
  const logoutBtn = $('logout-btn');

  const authGate = $('auth-gate');
  const authForm = $('auth-form');
  const authTitle = $('auth-title');
  const authSubtitle = $('auth-subtitle');
  const authEmail = $('auth-email');
  const authPassword = $('auth-password');
  const authSubmit = $('auth-submit');
  const authError = $('auth-error');
  const authTabLogin = $('auth-tab-login');
  const authTabSignup = $('auth-tab-signup');
  const authSwitch = $('auth-switch');

  window.handleMedbriefUnauthorized = () => {
    authSession = { authenticated: false, user: null, mode: 'signed-out' };
    currentConversationId = null;
    runtimeConfig = null;
    currentApiKeyId = '';
    memory.clearConversationCache();
    clearApiKeyVisualState();
    renderSessionUi();
    renderConversationList();
    showWelcomeView();
    showAuthGate('Your session expired. Sign in again to keep chatting.');
    setStatus('ready');
  };

  async function init() {
    applyTheme(currentTheme);
    setupEventListeners();
    autoResizeInput();
    loadPreferences();
    renderConversationList();
    await bootstrapSession();

    if (window.innerWidth <= 768) {
      sidebar.classList.add('collapsed');
    }

    if (authSession.authenticated) {
      messageInput.focus();
    }
  }

  async function bootstrapSession() {
    try {
      const response = await fetch(apiUrl('/api/auth/session'), { credentials: 'include' });
      if (!response.ok) {
        throw new Error('Unable to load session');
      }
      authSession = await response.json();
    } catch {
      authSession = { authenticated: false, user: null, mode: 'signed-out' };
    }

    renderSessionUi();
    if (!authSession.authenticated) {
      memory.clearConversationCache();
      renderConversationList();
      showAuthGate();
      showWelcomeView();
      return;
    }

    hideAuthGate();
    await loadRuntimeConfig();
    clearApiKeyVisualState();
    await syncConversationsFromServer();
    await loadSavedApiKey();
    showWelcomeView();
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    currentTheme = theme;
    localStorage.setItem('recall_theme', theme);

    if (theme === 'dark') {
      themeIconDark.style.display = '';
      themeIconLight.style.display = 'none';
    } else {
      themeIconDark.style.display = 'none';
      themeIconLight.style.display = '';
    }
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', handleSend);

    messageInput.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSend();
      }
    });

    messageInput.addEventListener('input', () => {
      autoResizeInput();
      sendBtn.disabled = !messageInput.value.trim();
    });

    newChatBtn.addEventListener('click', startNewConversation);

    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });

    themeToggle.addEventListener('click', () => {
      applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });

    moodCheckBtn.addEventListener('click', () => {
      moodModal.style.display = 'flex';
    });
    moodClose.addEventListener('click', () => {
      moodModal.style.display = 'none';
    });
    moodModal.addEventListener('click', event => {
      if (event.target === moodModal) moodModal.style.display = 'none';
    });

    document.querySelectorAll('.mood-option').forEach(btn => {
      btn.addEventListener('click', () => handleMoodSelect(btn));
    });

    crisisBtn.addEventListener('click', () => {
      crisisModal.style.display = 'flex';
    });
    crisisClose.addEventListener('click', () => {
      crisisModal.style.display = 'none';
    });
    crisisModal.addEventListener('click', event => {
      if (event.target === crisisModal) crisisModal.style.display = 'none';
    });

    settingsBtn.addEventListener('click', () => {
      populateSettingsPanel();
      settingsModal.style.display = 'flex';
    });
    settingsClose.addEventListener('click', () => {
      settingsModal.style.display = 'none';
    });
    settingsModal.addEventListener('click', event => {
      if (event.target === settingsModal) settingsModal.style.display = 'none';
    });

    if (generateApiKeyBtn) generateApiKeyBtn.addEventListener('click', handleGenerateApiKey);
    if (revokeApiKeyBtn) revokeApiKeyBtn.addEventListener('click', handleRevokeApiKey);
    if (apiKeyCopy) apiKeyCopy.addEventListener('click', handleCopyApiKey);
    clearMemoryBtn.addEventListener('click', handleClearMemory);

    prefMaxTokens.addEventListener('change', () => {
      const value = Math.max(50, Math.min(2000, parseInt(prefMaxTokens.value, 10) || 150));
      prefMaxTokens.value = value;
      RECALL_CONFIG.MAX_TOKENS = value;
      localStorage.setItem('recall_max_tokens', value);
    });

    prefTemperature.addEventListener('change', () => {
      const value = Math.max(0, Math.min(2, parseFloat(prefTemperature.value) || 0.8));
      prefTemperature.value = value;
      RECALL_CONFIG.TEMPERATURE = value;
      localStorage.setItem('recall_temperature', value);
    });

    welcomePrompts.addEventListener('click', event => {
      const card = event.target.closest('.prompt-card');
      if (!card) return;
      const prompt = card.dataset.prompt || '';
      if (!authSession.authenticated) {
        showAuthGate('Create an account or sign in to start chatting.');
        return;
      }
      messageInput.value = prompt;
      autoResizeInput();
      sendBtn.disabled = false;
      handleSend();
    });

    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        moodModal.style.display = 'none';
        crisisModal.style.display = 'none';
        settingsModal.style.display = 'none';
      }
    });

    authTabLogin.addEventListener('click', () => setAuthMode('login'));
    authTabSignup.addEventListener('click', () => setAuthMode('signup'));
    authSwitch.addEventListener('click', () => setAuthMode(authMode === 'login' ? 'signup' : 'login'));
    authForm.addEventListener('submit', handleAuthSubmit);
    logoutBtn.addEventListener('click', handleLogout);
  }

  function setAuthMode(mode) {
    authMode = mode === 'signup' ? 'signup' : 'login';
    authTabLogin.classList.toggle('active', authMode === 'login');
    authTabSignup.classList.toggle('active', authMode === 'signup');
    authTitle.textContent = authMode === 'login' ? 'Welcome back' : 'Create your account';
    authSubtitle.textContent = authMode === 'login'
      ? 'Sign in to access your saved MedBrief conversations from anywhere.'
      : 'Create an account to save your chats and keep them available worldwide.';
    authSubmit.textContent = authMode === 'login' ? 'Sign In' : 'Create Account';
    authSwitch.textContent = authMode === 'login' ? 'Need an account? Create one' : 'Already have an account? Sign in';
    setAuthError('');
  }

  function showAuthGate(message = '') {
    setAuthMode(authMode);
    authGate.style.display = 'flex';
    if (message) setAuthError(message);
    authEmail.focus();
  }

  function hideAuthGate() {
    authGate.style.display = 'none';
    setAuthError('');
    authPassword.value = '';
  }

  function setAuthError(message) {
    authError.textContent = message || '';
    authError.style.display = message ? 'block' : 'none';
  }

  function renderSessionUi() {
    const email = authSession?.user?.email || 'Signed out';
    sessionEmail.textContent = email;
    sessionMode.textContent = authSession?.mode === 'local-dev' ? 'Local dev session' : 'Account session';
    sessionPill.textContent = authSession?.authenticated
      ? (authSession.mode === 'local-dev' ? 'Local Dev' : 'Signed In')
      : 'Signed Out';
    logoutBtn.style.display = authSession?.authenticated && authSession.mode !== 'local-dev' ? 'inline-flex' : 'none';
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    const email = authEmail.value.trim();
    const password = authPassword.value;
    if (!email || !password) {
      setAuthError('Enter both an email address and password.');
      return;
    }

    authSubmit.disabled = true;
    authSubmit.textContent = authMode === 'login' ? 'Signing In…' : 'Creating…';
    setAuthError('');

    try {
      const response = await fetch(apiUrl(authMode === 'login' ? '/api/auth/login' : '/api/auth/signup'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || 'Authentication failed.');
      }
      authSession = payload;
      renderSessionUi();
      hideAuthGate();
      currentConversationId = null;
      memory.clearConversationCache();
      await loadRuntimeConfig();
      clearApiKeyVisualState();
      await syncConversationsFromServer();
      await loadSavedApiKey();
      showWelcomeView();
      messageInput.focus();
    } catch (error) {
      setAuthError(error.message || 'Authentication failed.');
    } finally {
      authSubmit.disabled = false;
      authSubmit.textContent = authMode === 'login' ? 'Sign In' : 'Create Account';
    }
  }

  async function handleLogout() {
    try {
      await fetch(apiUrl('/api/auth/logout'), {
        method: 'POST',
        credentials: 'include'
      });
    } catch {}
    window.handleMedbriefUnauthorized();
  }

  async function apiFetch(path, options = {}) {
    const response = await fetch(apiUrl(path), {
      credentials: 'include',
      ...options
    });
    if (response.status === 401) {
      window.handleMedbriefUnauthorized();
      throw new Error('Authentication required');
    }
    return response;
  }

  async function syncConversationsFromServer() {
    if (!authSession.authenticated) return;
    try {
      const response = await apiFetch('/api/conversations');
      if (!response.ok) throw new Error('Unable to load conversations');
      const payload = await response.json();
      memory.replaceConversations(payload.data || []);
      renderConversationList();
    } catch (error) {
      console.error('Conversation sync failed:', error);
    }
  }

  function autoResizeInput() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
  }

  async function handleSend() {
    const text = messageInput.value.trim();
    if (!text || engine.isGenerating) return;
    if (!authSession.authenticated) {
      showAuthGate('Create an account or sign in to send messages.');
      return;
    }

    if (!currentConversationId) {
      const conv = memory.createConversation();
      currentConversationId = conv.id;
      renderConversationList();
    }

    showChatView();
    appendUserMessage(text);

    messageInput.value = '';
    autoResizeInput();
    sendBtn.disabled = true;

    setStatus('thinking');
    showTypingIndicator();

    engine.sendMessage(
      currentConversationId,
      text,
      chunk => {
        hideTypingIndicator();
        appendAssistantChunk(chunk);
        scrollToBottom();
      },
      fullContent => {
        finalizeAssistantMessage(fullContent);
        setStatus('ready');
        renderConversationList();
        scrollToBottom();
      },
      error => {
        hideTypingIndicator();
        setStatus('ready');
        if (error.message !== 'Authentication required') {
          appendSystemMessage('Something went wrong. Please try again.');
        }
      }
    );
  }

  function startNewConversation() {
    if (!authSession.authenticated) {
      showAuthGate('Sign in to start a conversation.');
      return;
    }
    currentConversationId = null;
    showWelcomeView();
    renderConversationList();
    messageInput.value = '';
    autoResizeInput();
    sendBtn.disabled = true;
    messageInput.focus();
  }

  function showChatView() {
    welcomeScreen.style.display = 'none';
    chatMessages.style.display = 'flex';
    chatMessages.style.flexDirection = 'column';
    chatMessages.style.flex = '1';
  }

  function showWelcomeView() {
    welcomeScreen.style.display = 'flex';
    chatMessages.style.display = 'none';
    messagesContainer.innerHTML = '';
  }

  function appendUserMessage(text) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgEl = document.createElement('div');
    msgEl.className = 'message user';
    msgEl.innerHTML = `
      <div class="message-avatar">You</div>
      <div>
        <div class="message-content"><p>${escapeHtml(text)}</p></div>
        <div class="message-meta"><span class="message-time">${timeStr}</span></div>
      </div>
    `;
    messagesContainer.appendChild(msgEl);
    scrollToBottom();
  }

  let currentAssistantEl = null;
  let currentAssistantContent = '';

  function appendAssistantChunk(chunk) {
    if (!currentAssistantEl) {
      currentAssistantContent = '';
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      currentAssistantEl = document.createElement('div');
      currentAssistantEl.className = 'message assistant';
      currentAssistantEl.innerHTML = `
        <div class="message-avatar">
          <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
            <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
            <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="message-content"></div>
          <div class="message-meta"><span class="message-time">${timeStr}</span></div>
        </div>
      `;
      messagesContainer.appendChild(currentAssistantEl);
    }

    currentAssistantContent += chunk;
    const contentEl = currentAssistantEl.querySelector('.message-content');
    contentEl.innerHTML = engine.parseMarkdown(currentAssistantContent);
  }

  function finalizeAssistantMessage(fullContent) {
    if (currentAssistantEl) {
      const contentEl = currentAssistantEl.querySelector('.message-content');
      contentEl.innerHTML = engine.parseMarkdown(fullContent);

      if (engine.detectCrisis(fullContent) || containsCrisisResources(fullContent)) {
        addSafetyBanner(currentAssistantEl);
      }
    }
    currentAssistantEl = null;
    currentAssistantContent = '';
  }

  function containsCrisisResources(text) {
    return text.includes('988') && (text.includes('crisis') || text.includes('Crisis'));
  }

  function addSafetyBanner(messageEl) {
    const banner = document.createElement('div');
    banner.className = 'safety-banner';
    banner.innerHTML = `<p>MedBrief AI cares about your safety. If you need immediate help, please call <a href="tel:988"><strong>988</strong></a> or <a href="tel:911"><strong>911</strong></a>.</p>`;
    messageEl.querySelector('.message-content').appendChild(banner);
  }

  function showTypingIndicator() {
    hideTypingIndicator();
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
      <div class="message-avatar">
        <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
          <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    messagesContainer.appendChild(indicator);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    const existing = document.getElementById('typing-indicator');
    if (existing) existing.remove();
  }

  function appendSystemMessage(text) {
    const el = document.createElement('div');
    el.className = 'session-divider';
    el.innerHTML = `<span>${escapeHtml(text)}</span>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  function setStatus(status) {
    const dot = headerStatus.querySelector('.status-dot');
    const span = headerStatus.querySelector('span');

    if (status === 'thinking') {
      dot.style.background = 'var(--accent-primary)';
      span.textContent = 'MedBrief AI is thinking…';
    } else {
      dot.style.background = 'var(--success)';
      span.textContent = 'MedBrief AI is ready';
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function renderConversationList() {
    const convs = memory.getAllConversations();
    conversationList.innerHTML = '';

    const label = document.createElement('div');
    label.className = 'sidebar-section-label';
    label.textContent = authSession.authenticated ? 'Recent' : 'Sign in to save chats';
    conversationList.appendChild(label);

    convs.forEach(conv => {
      const item = document.createElement('div');
      item.className = 'conversation-item' + (conv.id === currentConversationId ? ' active' : '');
      item.innerHTML = `
        <span class="conv-title">${escapeHtml(conv.title)}</span>
        <button class="conv-delete" data-id="${conv.id}" title="Delete conversation">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2.5 2.5L9.5 9.5M9.5 2.5L2.5 9.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
          </svg>
        </button>
      `;

      item.addEventListener('click', event => {
        if (event.target.closest('.conv-delete')) return;
        loadConversation(conv.id);
      });

      item.querySelector('.conv-delete').addEventListener('click', async event => {
        event.stopPropagation();
        await deleteConversation(conv.id);
      });

      conversationList.appendChild(item);
    });
  }

  async function loadConversation(id) {
    let conv = memory.getConversation(id);
    if (!conv) return;

    currentConversationId = id;

    if (authSession.authenticated && conv.serverId) {
      try {
        const response = await apiFetch(`/api/conversations/${encodeURIComponent(conv.serverId)}`);
        if (response.ok) {
          const payload = await response.json();
          conv = memory.hydrateServerConversation(payload, id);
        }
      } catch (error) {
        console.error('Failed to load server conversation:', error);
      }
    }

    showChatView();
    messagesContainer.innerHTML = '';

    (conv.messages || []).forEach(msg => {
      if (msg.role === 'user') {
        appendExistingUserMessage(msg);
      } else if (msg.role === 'assistant') {
        appendExistingAssistantMessage(msg);
      }
    });

    renderConversationList();
    scrollToBottom();

    if (window.innerWidth <= 768) {
      sidebar.classList.add('collapsed');
    }
  }

  function appendExistingUserMessage(msg) {
    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const el = document.createElement('div');
    el.className = 'message user';
    el.innerHTML = `
      <div class="message-avatar">You</div>
      <div>
        <div class="message-content"><p>${escapeHtml(msg.content)}</p></div>
        <div class="message-meta"><span class="message-time">${time}</span></div>
      </div>
    `;
    messagesContainer.appendChild(el);
  }

  function appendExistingAssistantMessage(msg) {
    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
      <div class="message-avatar">
        <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
          <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="message-content">${engine.parseMarkdown(msg.content)}</div>
        <div class="message-meta"><span class="message-time">${time}</span></div>
      </div>
    `;
    messagesContainer.appendChild(el);
  }

  async function deleteConversation(id) {
    const conv = memory.getConversation(id);
    if (!conv) return;

    if (authSession.authenticated && conv.serverId) {
      try {
        const response = await apiFetch(`/api/conversations/${encodeURIComponent(conv.serverId)}`, {
          method: 'DELETE'
        });
        if (!response.ok) throw new Error('Delete failed');
      } catch {
        appendSystemMessage('Unable to delete that conversation right now.');
        return;
      }
    }

    memory.deleteConversation(id);
    if (currentConversationId === id) {
      startNewConversation();
    }
    renderConversationList();
  }

  function handleMoodSelect(btn) {
    document.querySelectorAll('.mood-option').forEach(node => node.classList.remove('selected'));
    btn.classList.add('selected');

    const value = parseInt(btn.dataset.value, 10);
    memory.recordMood(value);

    setTimeout(() => {
      moodModal.style.display = 'none';
      btn.classList.remove('selected');

      const moodLabel = RECALL_CONFIG.MOOD_LABELS[value];
      const moodResponses = {
        5: "It's wonderful to hear you're feeling good. What's contributing to that today?",
        4: "Glad you're holding steady. Anything specific making today feel okay?",
        3: "A meh day still counts. Is there anything in particular keeping you in the middle?",
        2: "I'm sorry you're feeling low. Would it help to talk about what's pulling you down?",
        1: "I hear you. It takes courage to say when things feel hard. I'm here. What feels heaviest right now?"
      };

      if (!authSession.authenticated) {
        showAuthGate('Create an account or sign in to save mood check-ins and conversations.');
        return;
      }

      if (!currentConversationId) {
        const conv = memory.createConversation();
        currentConversationId = conv.id;
        memory.updateConversation(conv.id, { title: `Mood check-in: ${moodLabel}` });
      }

      showChatView();
      memory.addMessage(currentConversationId, 'user', `I just did a mood check-in. I'm feeling ${moodLabel} right now.`);
      appendUserMessage(`I just did a mood check-in. I'm feeling ${moodLabel} right now.`);

      const response = moodResponses[value];
      memory.addMessage(currentConversationId, 'assistant', response);

      currentAssistantContent = '';
      currentAssistantEl = null;

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const msgEl = document.createElement('div');
      msgEl.className = 'message assistant';
      msgEl.innerHTML = `
        <div class="message-avatar">
          <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
            <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
            <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="message-content">${engine.parseMarkdown(response)}</div>
          <div class="message-meta"><span class="message-time">${timeStr}</span></div>
        </div>
      `;
      messagesContainer.appendChild(msgEl);

      renderConversationList();
      scrollToBottom();
    }, 500);
  }

  function populateSettingsPanel() {
    $('runtime-model').textContent = runtimeConfig?.active_model || runtimeConfig?.model_id || RECALL_CONFIG.MODEL || '—';
    $('runtime-endpoint').textContent = apiUrl('/v1/chat/completions');
    $('runtime-stream').textContent = RECALL_CONFIG.STREAM ? 'Enabled' : 'Disabled';
    $('runtime-temp').textContent = RECALL_CONFIG.TEMPERATURE || '—';

    $('mem-conversations').textContent = memory.getConversationCount();
    $('mem-moods').textContent = (memory.moods || []).length;
    $('mem-themes').textContent = (memory.profile?.themes || []).length;

    prefMaxTokens.value = RECALL_CONFIG.MAX_TOKENS;
    prefTemperature.value = RECALL_CONFIG.TEMPERATURE;
    sessionEmail.textContent = authSession?.user?.email || 'Signed out';
    sessionMode.textContent = authSession?.mode === 'local-dev' ? 'Local dev session' : 'Account session';
  }

  async function loadRuntimeConfig() {
    try {
      const response = await apiFetch('/api/config');
      if (!response.ok) return;
      runtimeConfig = await response.json();
      RECALL_CONFIG.MODEL = runtimeConfig.active_model || runtimeConfig.model_id || RECALL_CONFIG.MODEL;
      RECALL_CONFIG.API_ENDPOINT = apiUrl('/api/chat/completions');

      if (!localStorage.getItem('recall_max_tokens') && runtimeConfig?.default_generation?.max_new_tokens) {
        RECALL_CONFIG.MAX_TOKENS = runtimeConfig.default_generation.max_new_tokens;
      }
      if (!localStorage.getItem('recall_temperature') && typeof runtimeConfig?.default_generation?.temperature === 'number') {
        RECALL_CONFIG.TEMPERATURE = runtimeConfig.default_generation.temperature;
      }

      prefMaxTokens.value = RECALL_CONFIG.MAX_TOKENS;
      prefTemperature.value = RECALL_CONFIG.TEMPERATURE;

      if (developerSection) {
        developerSection.style.display = runtimeConfig.api_key_self_serve_enabled ? '' : 'none';
      }
    } catch {
      RECALL_CONFIG.API_ENDPOINT = apiUrl('/api/chat/completions');
      if (developerSection) developerSection.style.display = 'none';
    }
  }

  async function loadSavedApiKey() {
    if (!runtimeConfig?.api_key_self_serve_enabled) {
      clearApiKeyVisualState();
      return;
    }

    const savedKey = localStorage.getItem('recall_api_key');
    if (savedKey && savedKey !== 'local-model') {
      RECALL_CONFIG.API_KEY = savedKey;
      apiKeyValue.textContent = savedKey;
      apiKeyDisplay.style.display = 'block';
      revokeApiKeyBtn.style.display = 'inline-flex';
      generateApiKeyBtn.textContent = 'Regenerate API Key';
    }

    try {
      const response = await apiFetch('/api/keys');
      if (!response.ok) return;
      const payload = await response.json();
      const activeKey = (payload.data || []).find(key => !key.revoked_at) || null;
      if (!activeKey) {
        clearApiKeyVisualState();
        currentApiKeyId = '';
        localStorage.removeItem('recall_api_key_id');
        return;
      }
      currentApiKeyId = activeKey.id;
      localStorage.setItem('recall_api_key_id', currentApiKeyId);
      if (!savedKey || savedKey === 'local-model') {
        apiKeyValue.textContent = activeKey.key_prefix;
        apiKeyDisplay.style.display = 'block';
      }
      revokeApiKeyBtn.style.display = 'inline-flex';
      generateApiKeyBtn.textContent = 'Regenerate API Key';
    } catch {}
  }

  function clearApiKeyVisualState() {
    localStorage.removeItem('recall_api_key');
    localStorage.removeItem('recall_api_key_id');
    RECALL_CONFIG.API_KEY = 'local-model';
    apiKeyValue.textContent = '';
    apiKeyDisplay.style.display = 'none';
    revokeApiKeyBtn.style.display = 'none';
    if (generateApiKeyBtn) generateApiKeyBtn.textContent = 'Generate API Key';
  }

  async function handleGenerateApiKey() {
    try {
      const response = await apiFetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: 'MedBrief UI key' })
      });
      if (!response.ok) throw new Error('Key generation failed');
      const payload = await response.json();
      const key = payload.api_key || '';
      currentApiKeyId = payload.record?.id || '';
      if (currentApiKeyId) localStorage.setItem('recall_api_key_id', currentApiKeyId);
      localStorage.setItem('recall_api_key', key);
      RECALL_CONFIG.API_KEY = key;
      apiKeyValue.textContent = key;
      apiKeyDisplay.style.display = 'block';
      revokeApiKeyBtn.style.display = 'inline-flex';
      generateApiKeyBtn.textContent = 'Regenerate API Key';
    } catch {
      appendSystemMessage('Unable to generate an API key right now.');
    }
  }

  async function handleRevokeApiKey() {
    if (!currentApiKeyId) {
      clearApiKeyVisualState();
      return;
    }
    try {
      const response = await apiFetch(`/api/keys/${currentApiKeyId}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Key revoke failed');
      currentApiKeyId = '';
      clearApiKeyVisualState();
    } catch {
      appendSystemMessage('Unable to revoke the API key right now.');
    }
  }

  function handleCopyApiKey() {
    const key = apiKeyValue.textContent;
    if (!key) return;
    navigator.clipboard.writeText(key).then(() => {
      const original = apiKeyCopy.innerHTML;
      apiKeyCopy.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 7L6 10L11 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      setTimeout(() => { apiKeyCopy.innerHTML = original; }, 1500);
    });
  }

  async function handleClearMemory() {
    if (!confirm('This will permanently delete your saved conversations on this account. Continue?')) return;

    try {
      const conversations = memory.getAllConversations();
      for (const conv of conversations) {
        if (!conv.serverId) continue;
        await apiFetch(`/api/conversations/${encodeURIComponent(conv.serverId)}`, { method: 'DELETE' });
      }
      await apiFetch('/api/profile/memory', { method: 'DELETE' });
    } catch {
      appendSystemMessage('Unable to clear all saved data right now.');
      return;
    }

    memory.clearConversationCache();
    currentConversationId = null;
    renderConversationList();
    showWelcomeView();
    populateSettingsPanel();
  }

  function loadPreferences() {
    const savedTokens = localStorage.getItem('recall_max_tokens');
    const savedTemp = localStorage.getItem('recall_temperature');
    if (savedTokens) {
      RECALL_CONFIG.MAX_TOKENS = parseInt(savedTokens, 10);
      prefMaxTokens.value = savedTokens;
    }
    if (savedTemp) {
      RECALL_CONFIG.TEMPERATURE = parseFloat(savedTemp);
      prefTemperature.value = savedTemp;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  init();
});
