class RecallEngine {
  constructor(memory) {
    this.memory = memory;
    this.abortController = null;
    this.isGenerating = false;
  }

  detectCrisis(text) {
    const lower = text
      .toLowerCase()
      .replace(/’/g, "'")
      .replace(/[^a-z0-9'\s]/g, ' ')
      .replace(/([a-z])\1{2,}/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
    return RECALL_CONFIG.CRISIS_KEYWORDS.some(kw => lower.includes(kw));
  }

  apiUrl(path) {
    return `${RECALL_CONFIG.API_BASE || ''}${path}`;
  }

  handleUnauthorized() {
    if (typeof window.handleMedbriefUnauthorized === 'function') {
      window.handleMedbriefUnauthorized();
    }
  }

  async ensureServerConversation(conversationId) {
    const conversation = this.memory.getConversation(conversationId);
    if (!conversation) {
      throw new Error('Conversation not found');
    }
    if (conversation.serverId) {
      return conversation.serverId;
    }

    const response = await fetch(this.apiUrl('/api/conversations'), {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        title: conversation.title
      })
    });

    if (response.status === 401) {
      this.handleUnauthorized();
      throw new Error('Authentication required');
    }
    if (!response.ok) {
      throw new Error(`Conversation create error: ${response.status}`);
    }

    const payload = await response.json();
    const serverId = payload.id || '';
    if (!serverId) {
      throw new Error('Conversation create error: missing id');
    }

    this.memory.hydrateServerConversation(payload, conversationId);

    return serverId;
  }

  async sendMessage(conversationId, userMessage, onChunk, onComplete, onError) {
    if (this.isGenerating) {
      this.cancel();
    }

    this.isGenerating = true;
    this.abortController = new AbortController();

    const isCrisis = this.detectCrisis(userMessage);

    this.memory.addMessage(conversationId, 'user', userMessage);

    try {
      const serverConversationId = await this.ensureServerConversation(conversationId);
      if (RECALL_CONFIG.STREAM) {
        await this.streamRequest(serverConversationId, conversationId, userMessage, isCrisis, onChunk, onComplete, onError);
      } else {
        await this.standardRequest(serverConversationId, conversationId, userMessage, isCrisis, onChunk, onComplete, onError);
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        this.isGenerating = false;
        return;
      }
      console.error('Engine error:', error);
      onError(error);
    }
  }

  async streamRequest(serverConversationId, conversationId, userMessage, isCrisis, onChunk, onComplete, onError) {
    const response = await fetch(this.apiUrl(`/api/conversations/${encodeURIComponent(serverConversationId)}/messages`), {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        content: userMessage,
        stream: true
      }),
      signal: this.abortController.signal
    });

    if (response.status === 401) {
      this.handleUnauthorized();
      throw new Error('Authentication required');
    }
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let buffer = '';
    let finalPayload = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;
        if (!trimmed.startsWith('data: ')) continue;

        try {
          const json = JSON.parse(trimmed.slice(6));
          if (json.type === 'delta' && typeof json.delta === 'string') {
            const delta = json.delta;
            fullContent += delta;
            onChunk(delta);
            continue;
          }
          if (json.type === 'done') {
            finalPayload = json;
          }
        } catch {
          continue;
        }
      }
    }

    const hydratedFromServer = Boolean(finalPayload?.conversation);
    let finalContent = String(finalPayload?.reply || fullContent || '').trim();
    if (hydratedFromServer) {
      this.memory.hydrateServerConversation(finalPayload.conversation, conversationId);
    }
    if (finalContent && fullContent && finalContent.startsWith(fullContent) && finalContent.length > fullContent.length) {
      onChunk(finalContent.slice(fullContent.length));
    } else if (finalContent && !fullContent) {
      onChunk(finalContent);
    }
    if (isCrisis && !fullContent.includes('988')) {
      finalContent += RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM;
      onChunk(RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM);
    }

    if (!hydratedFromServer) {
      this.memory.addMessage(conversationId, 'assistant', finalContent);
    }
    this.isGenerating = false;
    onComplete(finalContent);
  }

  async standardRequest(serverConversationId, conversationId, userMessage, isCrisis, onChunk, onComplete, onError) {
    const response = await fetch(this.apiUrl(`/api/conversations/${encodeURIComponent(serverConversationId)}/messages`), {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        content: userMessage,
        stream: false
      }),
      signal: this.abortController.signal
    });

    if (response.status === 401) {
      this.handleUnauthorized();
      throw new Error('Authentication required');
    }
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    const hydratedFromServer = Boolean(data.conversation);
    if (hydratedFromServer) {
      this.memory.hydrateServerConversation(data.conversation, conversationId);
    }
    let content = String(data.reply || data.message?.content || '').trim() || 'I wasn\'t able to form a clear response. Could you try rephrasing that?';

    if (isCrisis && !content.includes('988')) {
      content += RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM;
    }

    if (!hydratedFromServer) {
      this.memory.addMessage(conversationId, 'assistant', content);
    }
    this.isGenerating = false;
    onChunk(content);
    onComplete(content);
  }

  cancel() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isGenerating = false;
  }

  parseMarkdown(text) {
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');

    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    html = html.replace(/^---$/gm, '<hr>');

    const paragraphs = html.split(/\n\n+/).filter(p => p.trim());
    html = paragraphs.map(p => {
      p = p.trim();
      if (p.startsWith('<ul>') || p.startsWith('<hr>') || p.startsWith('<ol>')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
  }
}
