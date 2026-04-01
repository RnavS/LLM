class RecallMemory {
  constructor() {
    this.conversationsKey = 'recall_conversations';
    this.moodKey = 'recall_moods';
    this.profileKey = 'recall_profile';
    this.load();
  }

  load() {
    try {
      this.conversations = JSON.parse(localStorage.getItem(this.conversationsKey)) || [];
      this.moods = JSON.parse(localStorage.getItem(this.moodKey)) || [];
      this.profile = JSON.parse(localStorage.getItem(this.profileKey)) || this.defaultProfile();
    } catch {
      this.conversations = [];
      this.moods = [];
      this.profile = this.defaultProfile();
    }
  }

  defaultProfile() {
    return {
      themes: [],
      copingPrefs: [],
      recurringStressors: [],
      patterns: [],
      goals: [],
      createdAt: Date.now()
    };
  }

  save() {
    try {
      localStorage.setItem(this.conversationsKey, JSON.stringify(this.conversations));
      localStorage.setItem(this.moodKey, JSON.stringify(this.moods));
      localStorage.setItem(this.profileKey, JSON.stringify(this.profile));
    } catch (error) {
      console.warn('Storage save failed:', error);
    }
  }

  createConversation() {
    const conv = {
      id: this.generateId(),
      serverId: '',
      title: 'New conversation',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      preview: '',
      summary: ''
    };
    this.conversations.unshift(conv);
    this.save();
    return conv;
  }

  getConversation(id) {
    return this.conversations.find(c => c.id === id) || null;
  }

  getConversationByServerId(serverId) {
    return this.conversations.find(c => c.serverId === serverId || c.id === serverId) || null;
  }

  updateConversation(id, updates) {
    const conv = this.getConversation(id);
    if (conv) {
      Object.assign(conv, updates, { updatedAt: Date.now() });
      this.save();
    }
    return conv;
  }

  deleteConversation(id) {
    this.conversations = this.conversations.filter(c => c.id !== id);
    this.save();
  }

  removeConversationByServerId(serverId) {
    this.conversations = this.conversations.filter(c => c.serverId !== serverId && c.id !== serverId);
    this.save();
  }

  addMessage(convId, role, content) {
    const conv = this.getConversation(convId);
    if (!conv) return null;

    const msg = {
      id: this.generateId(),
      role,
      content,
      timestamp: Date.now()
    };
    conv.messages.push(msg);
    conv.updatedAt = Date.now();
    conv.preview = content.slice(0, 220);

    if (conv.messages.length === 1 && role === 'user') {
      conv.title = this.generateTitle(content);
    }

    this.save();
    return msg;
  }

  replaceConversations(serverConversations) {
    this.conversations = (serverConversations || []).map(conversation => this.toLocalConversation(conversation));
    this.save();
  }

  hydrateServerConversation(serverConversation, localId = '') {
    const normalized = this.toLocalConversation(serverConversation);
    const existing = (localId && this.getConversation(localId)) || this.getConversationByServerId(normalized.serverId);
    if (existing) {
      existing.serverId = normalized.serverId;
      existing.title = normalized.title;
      existing.createdAt = normalized.createdAt;
      existing.updatedAt = normalized.updatedAt;
      existing.preview = normalized.preview;
      if (normalized.messages.length > 0) {
        existing.messages = normalized.messages;
      }
      this.save();
      return existing;
    }
    this.conversations.unshift(normalized);
    this.save();
    return normalized;
  }

  toLocalConversation(serverConversation) {
    const createdAt = Date.parse(serverConversation.created_at || '') || Date.now();
    const updatedAt = Date.parse(serverConversation.updated_at || '') || createdAt;
    return {
      id: String(serverConversation.id || this.generateId()),
      serverId: String(serverConversation.id || ''),
      title: String(serverConversation.title || 'New conversation'),
      messages: (serverConversation.messages || []).map(message => this.toLocalMessage(message)),
      createdAt,
      updatedAt,
      preview: String(serverConversation.preview || ''),
      summary: String(serverConversation.preview || '')
    };
  }

  toLocalMessage(message) {
    return {
      id: String(message.id || this.generateId()),
      role: String(message.role || 'assistant'),
      content: String(message.content || ''),
      timestamp: Date.parse(message.created_at || '') || Date.now()
    };
  }

  generateTitle(content) {
    const cleaned = content.replace(/\n/g, ' ').trim();
    if (cleaned.length <= 40) return cleaned;
    const truncated = cleaned.substring(0, 40);
    const lastSpace = truncated.lastIndexOf(' ');
    return (lastSpace > 20 ? truncated.substring(0, lastSpace) : truncated) + '…';
  }

  getContextWindow(convId, maxMessages) {
    const conv = this.getConversation(convId);
    if (!conv) return [];
    return conv.messages.slice(-maxMessages).map(m => ({ role: m.role, content: m.content }));
  }

  buildMemoryContext() {
    const parts = [];

    if (this.profile.themes.length > 0) {
      parts.push(`Recurring themes: ${this.profile.themes.join(', ')}`);
    }
    if (this.profile.patterns.length > 0) {
      parts.push(`Identified patterns: ${this.profile.patterns.join('; ')}`);
    }
    if (this.profile.recurringStressors.length > 0) {
      parts.push(`Known stressors: ${this.profile.recurringStressors.join(', ')}`);
    }
    if (this.profile.goals.length > 0) {
      parts.push(`Active goals: ${this.profile.goals.join(', ')}`);
    }

    const recentMoods = this.moods.slice(-7);
    if (recentMoods.length > 0) {
      const moodSummary = recentMoods.map(m =>
        `${RECALL_CONFIG.MOOD_LABELS[m.value] || m.value} (${new Date(m.timestamp).toLocaleDateString()})`
      ).join(', ');
      parts.push(`Recent mood check-ins: ${moodSummary}`);
    }

    return parts.length > 0 ? parts.join('\n') : '';
  }

  recordMood(value) {
    this.moods.push({ value, timestamp: Date.now() });
    if (this.moods.length > 100) {
      this.moods = this.moods.slice(-100);
    }
    this.save();
  }

  addTheme(theme) {
    if (!this.profile.themes.includes(theme)) {
      this.profile.themes.push(theme);
      if (this.profile.themes.length > 15) this.profile.themes.shift();
      this.save();
    }
  }

  addPattern(pattern) {
    if (!this.profile.patterns.includes(pattern)) {
      this.profile.patterns.push(pattern);
      if (this.profile.patterns.length > 10) this.profile.patterns.shift();
      this.save();
    }
  }

  addStressor(stressor) {
    if (!this.profile.recurringStressors.includes(stressor)) {
      this.profile.recurringStressors.push(stressor);
      if (this.profile.recurringStressors.length > 10) this.profile.recurringStressors.shift();
      this.save();
    }
  }

  generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
  }

  getAllConversations() {
    return [...this.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  getConversationCount() {
    return this.conversations.length;
  }

  getMoodTrend(days = 7) {
    const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
    return this.moods.filter(m => m.timestamp >= cutoff);
  }

  clearConversationCache() {
    this.conversations = [];
    this.save();
  }

  clearAll() {
    this.conversations = [];
    this.moods = [];
    this.profile = this.defaultProfile();
    this.save();
  }
}
