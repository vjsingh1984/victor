/**
 * Inline Edit Provider
 *
 * Provides inline code editing with ghost text preview.
 * Shows AI-suggested changes directly in the editor before applying.
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

export interface InlineEditRequest {
    document: vscode.TextDocument;
    range: vscode.Range;
    instruction: string;
    context?: string;
}

export interface InlineEditResult {
    originalText: string;
    suggestedText: string;
    range: vscode.Range;
    explanation?: string;
}

/**
 * Decoration type for ghost text preview
 */
const ghostTextDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        color: new vscode.ThemeColor('editorGhostText.foreground'),
        fontStyle: 'italic',
    },
    isWholeLine: false,
});

/**
 * Decoration type for deletion highlight
 */
const deletionDecorationType = vscode.window.createTextEditorDecorationType({
    backgroundColor: new vscode.ThemeColor('diffEditor.removedTextBackground'),
    textDecoration: 'line-through',
});

/**
 * Decoration type for addition highlight
 */
const additionDecorationType = vscode.window.createTextEditorDecorationType({
    backgroundColor: new vscode.ThemeColor('diffEditor.insertedTextBackground'),
});

export class InlineEditProvider implements vscode.Disposable {
    private _disposables: vscode.Disposable[] = [];
    private _activeEdit?: InlineEditResult;
    private _activeEditor?: vscode.TextEditor;
    private _statusBarItem: vscode.StatusBarItem;
    private _isProcessing = false;

    constructor(
        private readonly _client: VictorClient,
        private readonly _log?: vscode.OutputChannel
    ) {
        // Create status bar item for inline edit status
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this._statusBarItem.text = '$(loading~spin) Victor Edit...';
        this._disposables.push(this._statusBarItem);

        // Register commands
        this._disposables.push(
            vscode.commands.registerCommand('victor.inlineEdit', () => this.startInlineEdit()),
            vscode.commands.registerCommand('victor.acceptInlineEdit', () => this.acceptEdit()),
            vscode.commands.registerCommand('victor.rejectInlineEdit', () => this.rejectEdit()),
        );

        // Listen for selection changes to offer inline edits
        this._disposables.push(
            vscode.window.onDidChangeTextEditorSelection(this._onSelectionChange.bind(this))
        );
    }

    public dispose(): void {
        this._clearDecorations();
        this._disposables.forEach(d => d.dispose());
    }

    /**
     * Start an inline edit for the current selection
     */
    public async startInlineEdit(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showWarningMessage('Please select code to edit');
            return;
        }

        // Get edit instruction from user
        const instruction = await vscode.window.showInputBox({
            prompt: 'What would you like to do with this code?',
            placeHolder: 'e.g., "fix the bug", "add error handling", "make it more efficient"',
            validateInput: (value) => value.trim() ? null : 'Please enter an instruction',
        });

        if (!instruction) {
            return; // User cancelled
        }

        await this._performInlineEdit({
            document: editor.document,
            range: selection,
            instruction,
        });
    }

    /**
     * Perform an inline edit with ghost text preview
     */
    private async _performInlineEdit(request: InlineEditRequest): Promise<void> {
        if (this._isProcessing) {
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document !== request.document) {
            return;
        }

        this._isProcessing = true;
        this._statusBarItem.show();
        this._log?.appendLine(`[InlineEdit] Starting edit: ${request.instruction}`);

        try {
            const originalText = request.document.getText(request.range);

            // Request edit from Victor
            const suggestedText = await this._requestEdit(
                originalText,
                request.instruction,
                request.document.languageId
            );

            if (!suggestedText || suggestedText === originalText) {
                vscode.window.showInformationMessage('No changes suggested');
                return;
            }

            // Store the active edit
            this._activeEdit = {
                originalText,
                suggestedText,
                range: request.range,
            };
            this._activeEditor = editor;

            // Show ghost text preview
            this._showEditPreview(editor, request.range, originalText, suggestedText);

            // Show accept/reject prompt
            const choice = await vscode.window.showInformationMessage(
                'Victor suggests changes. Accept?',
                { modal: false },
                'Accept (Tab)',
                'Reject (Esc)',
                'Show Diff'
            );

            if (choice === 'Accept (Tab)') {
                await this.acceptEdit();
            } else if (choice === 'Show Diff') {
                await this._showDiff(originalText, suggestedText, request.document.fileName);
            } else {
                this.rejectEdit();
            }

        } catch (error) {
            this._log?.appendLine(`[InlineEdit] Error: ${error}`);
            vscode.window.showErrorMessage(`Inline edit failed: ${error}`);
        } finally {
            this._isProcessing = false;
            this._statusBarItem.hide();
        }
    }

    /**
     * Request an edit from Victor
     */
    private async _requestEdit(
        code: string,
        instruction: string,
        language: string
    ): Promise<string> {
        const prompt = `Edit the following ${language} code according to this instruction: "${instruction}"

Code to edit:
\`\`\`${language}
${code}
\`\`\`

Return ONLY the edited code without any explanation. Do not include markdown code fences.`;

        let result = '';
        await this._client.streamChat(
            [{ role: 'user', content: prompt }],
            (chunk) => { result += chunk; },
            () => {} // Ignore tool calls for inline edit
        );

        // Clean up the result (remove markdown fences if present)
        return result
            .replace(/^```\w*\n/m, '')
            .replace(/\n```$/m, '')
            .trim();
    }

    /**
     * Show edit preview with ghost text decorations
     */
    private _showEditPreview(
        editor: vscode.TextEditor,
        range: vscode.Range,
        originalText: string,
        suggestedText: string
    ): void {
        // Clear existing decorations
        this._clearDecorations();

        // Show deletion decoration on original text
        editor.setDecorations(deletionDecorationType, [range]);

        // Calculate where to show the ghost text
        const endPosition = range.end;
        const ghostDecoration: vscode.DecorationOptions = {
            range: new vscode.Range(endPosition, endPosition),
            renderOptions: {
                after: {
                    contentText: `  → ${suggestedText.split('\n')[0]}${suggestedText.includes('\n') ? '...' : ''}`,
                    color: new vscode.ThemeColor('editorGhostText.foreground'),
                    fontStyle: 'italic',
                }
            }
        };

        editor.setDecorations(ghostTextDecorationType, [ghostDecoration]);
    }

    /**
     * Accept the current inline edit
     */
    public async acceptEdit(): Promise<void> {
        if (!this._activeEdit || !this._activeEditor) {
            return;
        }

        const edit = new vscode.WorkspaceEdit();
        edit.replace(
            this._activeEditor.document.uri,
            this._activeEdit.range,
            this._activeEdit.suggestedText
        );

        const success = await vscode.workspace.applyEdit(edit);
        if (success) {
            vscode.window.showInformationMessage('Edit applied');
            this._log?.appendLine('[InlineEdit] Edit accepted and applied');
        } else {
            vscode.window.showErrorMessage('Failed to apply edit');
        }

        this._clearActiveEdit();
    }

    /**
     * Reject the current inline edit
     */
    public rejectEdit(): void {
        this._clearActiveEdit();
        vscode.window.showInformationMessage('Edit rejected');
        this._log?.appendLine('[InlineEdit] Edit rejected');
    }

    /**
     * Show diff view for the suggested changes
     */
    private async _showDiff(
        originalText: string,
        suggestedText: string,
        fileName: string
    ): Promise<void> {
        // Create temporary URIs for diff
        const originalUri = vscode.Uri.parse(`victor-diff:original/${fileName}`);
        const suggestedUri = vscode.Uri.parse(`victor-diff:suggested/${fileName}`);

        // Register content provider (simplified - in production would use proper provider)
        const provider = new class implements vscode.TextDocumentContentProvider {
            provideTextDocumentContent(uri: vscode.Uri): string {
                if (uri.path.startsWith('original/')) {
                    return originalText;
                }
                return suggestedText;
            }
        };

        const disposable = vscode.workspace.registerTextDocumentContentProvider('victor-diff', provider);

        await vscode.commands.executeCommand('vscode.diff',
            originalUri,
            suggestedUri,
            `Victor Edit: ${fileName}`
        );

        // Clean up provider after a delay
        setTimeout(() => disposable.dispose(), 60000);
    }

    /**
     * Clear all decorations
     */
    private _clearDecorations(): void {
        if (this._activeEditor) {
            this._activeEditor.setDecorations(ghostTextDecorationType, []);
            this._activeEditor.setDecorations(deletionDecorationType, []);
            this._activeEditor.setDecorations(additionDecorationType, []);
        }
    }

    /**
     * Clear the active edit state
     */
    private _clearActiveEdit(): void {
        this._clearDecorations();
        this._activeEdit = undefined;
        this._activeEditor = undefined;
    }

    /**
     * Handle selection changes
     */
    private _onSelectionChange(event: vscode.TextEditorSelectionChangeEvent): void {
        // Clear preview if selection changes
        if (this._activeEdit && event.textEditor === this._activeEditor) {
            const currentSelection = event.selections[0];
            if (!currentSelection.isEqual(this._activeEdit.range)) {
                this._clearActiveEdit();
            }
        }
    }
}
