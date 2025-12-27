/**
 * Conversation Storage Tests
 */

import * as assert from 'assert';
import { StoredConversation, ConversationIndex } from '../../conversationStorage';

suite('ConversationStorage Test Suite', () => {
    test('StoredConversation interface', () => {
        const conversation: StoredConversation = {
            id: 'conv-123',
            title: 'Test Conversation',
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };

        assert.strictEqual(conversation.id, 'conv-123');
        assert.strictEqual(conversation.title, 'Test Conversation');
        assert.strictEqual(conversation.messages.length, 0);
    });

    test('StoredConversation with messages', () => {
        const conversation: StoredConversation = {
            id: 'conv-456',
            title: 'Chat with messages',
            messages: [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there!' },
            ],
            createdAt: Date.now() - 10000,
            updatedAt: Date.now(),
        };

        assert.strictEqual(conversation.messages.length, 2);
        assert.strictEqual(conversation.messages[0].role, 'user');
        assert.strictEqual(conversation.messages[1].role, 'assistant');
    });

    test('StoredConversation with metadata', () => {
        const conversation: StoredConversation = {
            id: 'conv-meta',
            title: 'Conversation with metadata',
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
            workspaceFolder: '/path/to/workspace',
            provider: 'anthropic',
            model: 'claude-3-opus',
            metadata: {
                tags: ['test', 'example'],
                priority: 'high',
            },
        };

        assert.strictEqual(conversation.workspaceFolder, '/path/to/workspace');
        assert.strictEqual(conversation.provider, 'anthropic');
        assert.strictEqual(conversation.model, 'claude-3-opus');
        assert.ok(conversation.metadata?.tags);
    });

    test('StoredConversation timestamps', () => {
        const created = Date.now() - 60000;
        const updated = Date.now();

        const conversation: StoredConversation = {
            id: 'conv-time',
            title: 'Timed conversation',
            messages: [],
            createdAt: created,
            updatedAt: updated,
        };

        assert.ok(conversation.updatedAt >= conversation.createdAt);
    });
});

suite('ConversationIndex Test Suite', () => {
    test('ConversationIndex structure', () => {
        const index: ConversationIndex = {
            version: 1,
            conversations: [],
        };

        assert.strictEqual(index.version, 1);
        assert.strictEqual(index.conversations.length, 0);
    });

    test('ConversationIndex with conversations', () => {
        const index: ConversationIndex = {
            version: 1,
            conversations: [
                {
                    id: 'conv-1',
                    title: 'First conversation',
                    messageCount: 5,
                    createdAt: Date.now() - 3600000,
                    updatedAt: Date.now() - 1800000,
                    preview: 'What is TypeScript?',
                },
                {
                    id: 'conv-2',
                    title: 'Second conversation',
                    messageCount: 10,
                    createdAt: Date.now() - 7200000,
                    updatedAt: Date.now(),
                    preview: 'Help me debug...',
                },
            ],
        };

        assert.strictEqual(index.conversations.length, 2);
        assert.strictEqual(index.conversations[0].messageCount, 5);
        assert.strictEqual(index.conversations[1].messageCount, 10);
    });

    test('ConversationIndex preview truncation', () => {
        const longMessage = 'A'.repeat(200);
        const preview = longMessage.length <= 100 ? longMessage : longMessage.slice(0, 97) + '...';

        const index: ConversationIndex = {
            version: 1,
            conversations: [{
                id: 'conv-long',
                title: 'Long preview',
                messageCount: 1,
                createdAt: Date.now(),
                updatedAt: Date.now(),
                preview,
            }],
        };

        assert.strictEqual(index.conversations[0].preview?.length, 100);
    });
});

suite('Conversation ID Generation', () => {
    test('ID format', () => {
        const id = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

        assert.ok(id.startsWith('conv-'));
        assert.ok(id.length > 15);
    });

    test('IDs are unique', () => {
        const ids = new Set<string>();
        for (let i = 0; i < 100; i++) {
            const id = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
            assert.ok(!ids.has(id), 'Generated duplicate ID');
            ids.add(id);
        }
        assert.strictEqual(ids.size, 100);
    });
});

suite('Title Generation', () => {
    test('Default title format', () => {
        const title = `Conversation ${new Date().toLocaleDateString()}`;
        assert.ok(title.startsWith('Conversation'));
    });

    test('Title from message (short)', () => {
        const content = 'How do I use TypeScript?';
        const title = content.length <= 50 ? content : content.slice(0, 47) + '...';

        assert.strictEqual(title, content);
        assert.ok(title.length <= 50);
    });

    test('Title from message (long)', () => {
        const content = 'A'.repeat(100);
        const title = content.length <= 50 ? content : content.slice(0, 47) + '...';

        assert.strictEqual(title.length, 50);
        assert.ok(title.endsWith('...'));
    });

    test('Title from multiline message', () => {
        const content = 'First line\nSecond line\nThird line';
        const firstLine = content.split('\n')[0];
        const title = firstLine.length <= 50 ? firstLine : firstLine.slice(0, 47) + '...';

        assert.strictEqual(title, 'First line');
    });
});

suite('Conversation Export', () => {
    test('Markdown export format', () => {
        const conversation: StoredConversation = {
            id: 'conv-export',
            title: 'Test Export',
            messages: [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi!' },
            ],
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };

        // Simulate markdown conversion
        let md = `# ${conversation.title}\n\n`;
        md += `_Created: ${new Date(conversation.createdAt).toLocaleString()}_\n\n`;
        md += `---\n\n`;

        for (const message of conversation.messages) {
            const roleLabel = message.role === 'user' ? '**You**' : '**Victor**';
            md += `### ${roleLabel}\n\n`;
            md += message.content + '\n\n';
        }

        assert.ok(md.includes('# Test Export'));
        assert.ok(md.includes('**You**'));
        assert.ok(md.includes('**Victor**'));
        assert.ok(md.includes('Hello'));
        assert.ok(md.includes('Hi!'));
    });
});

suite('Message Roles', () => {
    test('Valid message roles', () => {
        const roles: Array<'user' | 'assistant' | 'system'> = ['user', 'assistant', 'system'];

        roles.forEach(role => {
            const message = { role, content: 'Test' };
            assert.strictEqual(message.role, role);
        });
    });

    test('User message structure', () => {
        const message = {
            role: 'user' as const,
            content: 'Hello, Victor!',
        };

        assert.strictEqual(message.role, 'user');
        assert.ok(message.content.length > 0);
    });

    test('Assistant message with tool calls', () => {
        const message = {
            role: 'assistant' as const,
            content: 'Let me read that file.',
            toolCalls: [
                {
                    id: 'tc-1',
                    name: 'read_file',
                    status: 'completed' as const,
                    input: { path: '/test.txt' },
                    output: 'File contents',
                },
            ],
        };

        assert.strictEqual(message.role, 'assistant');
        assert.strictEqual(message.toolCalls?.length, 1);
        assert.strictEqual(message.toolCalls[0].name, 'read_file');
    });
});

suite('Storage Limits', () => {
    const MAX_CONVERSATIONS = 100;
    const MAX_PREVIEW_LENGTH = 100;

    test('Max conversations limit', () => {
        const conversations = Array.from({ length: 150 }, (_, i) => ({
            id: `conv-${i}`,
            title: `Conversation ${i}`,
            messageCount: i,
            createdAt: Date.now() - i * 1000,
            updatedAt: Date.now() - i * 1000,
        }));

        // Simulate trimming
        const trimmed = conversations.slice(0, MAX_CONVERSATIONS);
        assert.strictEqual(trimmed.length, MAX_CONVERSATIONS);
    });

    test('Max preview length', () => {
        const longPreview = 'A'.repeat(200);
        const trimmed = longPreview.slice(0, MAX_PREVIEW_LENGTH - 3) + '...';

        assert.strictEqual(trimmed.length, MAX_PREVIEW_LENGTH);
    });
});
