/**
 * Code Actions Provider
 *
 * Provides VS Code code actions (quick fixes) powered by Victor AI:
 * - Fix errors and warnings
 * - Refactor code
 * - Generate documentation
 * - Optimize code
 * - Add error handling
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

/**
 * Victor AI Code Action Provider
 */
export class VictorCodeActionProvider implements vscode.CodeActionProvider {
    public static readonly providedCodeActionKinds = [
        vscode.CodeActionKind.QuickFix,
        vscode.CodeActionKind.Refactor,
        vscode.CodeActionKind.RefactorExtract,
        vscode.CodeActionKind.RefactorInline,
        vscode.CodeActionKind.Source,
    ];

    constructor(private readonly _client: VictorClient) {}

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];

        // Add actions for diagnostics (errors/warnings)
        for (const diagnostic of context.diagnostics) {
            actions.push(...this._createDiagnosticActions(document, diagnostic, range));
        }

        // Add general actions for selected code
        if (!range.isEmpty) {
            actions.push(...this._createSelectionActions(document, range));
        }

        // Add actions for current line
        actions.push(...this._createLineActions(document, range));

        return actions;
    }

    /**
     * Create code actions for a specific diagnostic
     */
    private _createDiagnosticActions(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic,
        range: vscode.Range
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];

        // Fix with Victor AI
        const fixAction = new vscode.CodeAction(
            `$(hubot) Fix with Victor: ${this._truncate(diagnostic.message, 50)}`,
            vscode.CodeActionKind.QuickFix
        );
        fixAction.command = {
            command: 'victor.fixDiagnostic',
            title: 'Fix with Victor',
            arguments: [document, diagnostic],
        };
        fixAction.diagnostics = [diagnostic];
        fixAction.isPreferred = true;
        actions.push(fixAction);

        // Explain error
        const explainAction = new vscode.CodeAction(
            `$(question) Explain error with Victor`,
            vscode.CodeActionKind.QuickFix
        );
        explainAction.command = {
            command: 'victor.explainDiagnostic',
            title: 'Explain Error',
            arguments: [document, diagnostic],
        };
        explainAction.diagnostics = [diagnostic];
        actions.push(explainAction);

        return actions;
    }

    /**
     * Create code actions for selected code
     */
    private _createSelectionActions(
        document: vscode.TextDocument,
        range: vscode.Range
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];

        // Refactor with Victor
        const refactorAction = new vscode.CodeAction(
            '$(hubot) Refactor with Victor',
            vscode.CodeActionKind.Refactor
        );
        refactorAction.command = {
            command: 'victor.refactorSelection',
            title: 'Refactor Selection',
            arguments: [document, range],
        };
        actions.push(refactorAction);

        // Extract function
        const extractAction = new vscode.CodeAction(
            '$(hubot) Extract to function',
            vscode.CodeActionKind.RefactorExtract
        );
        extractAction.command = {
            command: 'victor.extractFunction',
            title: 'Extract to Function',
            arguments: [document, range],
        };
        actions.push(extractAction);

        // Add documentation
        const docAction = new vscode.CodeAction(
            '$(hubot) Add documentation',
            vscode.CodeActionKind.Source
        );
        docAction.command = {
            command: 'victor.addDocumentation',
            title: 'Add Documentation',
            arguments: [document, range],
        };
        actions.push(docAction);

        // Optimize code
        const optimizeAction = new vscode.CodeAction(
            '$(hubot) Optimize code',
            vscode.CodeActionKind.Refactor
        );
        optimizeAction.command = {
            command: 'victor.optimizeCode',
            title: 'Optimize Code',
            arguments: [document, range],
        };
        actions.push(optimizeAction);

        // Add error handling
        const errorHandlingAction = new vscode.CodeAction(
            '$(hubot) Add error handling',
            vscode.CodeActionKind.Refactor
        );
        errorHandlingAction.command = {
            command: 'victor.addErrorHandling',
            title: 'Add Error Handling',
            arguments: [document, range],
        };
        actions.push(errorHandlingAction);

        return actions;
    }

    /**
     * Create code actions for current line
     */
    private _createLineActions(
        document: vscode.TextDocument,
        range: vscode.Range
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];
        const line = document.lineAt(range.start.line);

        // Check for TODO/FIXME comments
        if (/\b(TODO|FIXME|HACK|XXX)\b/i.test(line.text)) {
            const todoAction = new vscode.CodeAction(
                '$(hubot) Implement TODO with Victor',
                vscode.CodeActionKind.QuickFix
            );
            todoAction.command = {
                command: 'victor.implementTodo',
                title: 'Implement TODO',
                arguments: [document, line],
            };
            actions.push(todoAction);
        }

        return actions;
    }

    private _truncate(text: string, maxLength: number): string {
        if (text.length <= maxLength) {return text;}
        return text.substring(0, maxLength - 3) + '...';
    }
}

/**
 * Register code action commands
 */
export function registerCodeActionCommands(
    context: vscode.ExtensionContext,
    client: VictorClient,
    sendToChat: (message: string) => Promise<void>
): void {
    // Fix diagnostic with Victor
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.fixDiagnostic',
            async (document: vscode.TextDocument, diagnostic: vscode.Diagnostic) => {
                const code = document.getText(diagnostic.range);
                const contextLines = getContextLines(document, diagnostic.range, 5);

                const prompt = `Fix this error in ${document.languageId}:

Error: ${diagnostic.message}
${diagnostic.source ? `Source: ${diagnostic.source}` : ''}
${diagnostic.code ? `Code: ${diagnostic.code}` : ''}

Code with error:
\`\`\`${document.languageId}
${contextLines}
\`\`\`

The specific problematic code is: \`${code}\`

Please provide the corrected code.`;

                await sendToChat(prompt);
            }
        )
    );

    // Explain diagnostic
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.explainDiagnostic',
            async (document: vscode.TextDocument, diagnostic: vscode.Diagnostic) => {
                const code = document.getText(diagnostic.range);
                const contextLines = getContextLines(document, diagnostic.range, 3);

                const prompt = `Explain this ${document.languageId} error:

Error: ${diagnostic.message}
${diagnostic.source ? `Source: ${diagnostic.source}` : ''}
${diagnostic.code ? `Code: ${diagnostic.code}` : ''}

Code:
\`\`\`${document.languageId}
${contextLines}
\`\`\`

What causes this error and how can I fix it?`;

                await sendToChat(prompt);
            }
        )
    );

    // Refactor selection
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.refactorSelection',
            async (document: vscode.TextDocument, range: vscode.Range) => {
                const code = document.getText(range);

                const suggestion = await vscode.window.showInputBox({
                    prompt: 'What refactoring would you like?',
                    placeHolder: 'e.g., simplify, make more readable, use modern syntax',
                });

                if (!suggestion) {return;}

                const prompt = `Refactor this ${document.languageId} code (${suggestion}):

\`\`\`${document.languageId}
${code}
\`\`\`

Provide the refactored code.`;

                await sendToChat(prompt);
            }
        )
    );

    // Extract to function
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.extractFunction',
            async (document: vscode.TextDocument, range: vscode.Range) => {
                const code = document.getText(range);

                const functionName = await vscode.window.showInputBox({
                    prompt: 'Enter function name',
                    placeHolder: 'e.g., calculateTotal, handleSubmit',
                });

                if (!functionName) {return;}

                const prompt = `Extract this ${document.languageId} code into a function named "${functionName}":

\`\`\`${document.languageId}
${code}
\`\`\`

Provide the extracted function and show how to call it.`;

                await sendToChat(prompt);
            }
        )
    );

    // Add documentation
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.addDocumentation',
            async (document: vscode.TextDocument, range: vscode.Range) => {
                const code = document.getText(range);

                const prompt = `Add comprehensive documentation to this ${document.languageId} code:

\`\`\`${document.languageId}
${code}
\`\`\`

Include:
- Function/class docstrings
- Parameter descriptions
- Return value descriptions
- Usage examples if applicable`;

                await sendToChat(prompt);
            }
        )
    );

    // Optimize code
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.optimizeCode',
            async (document: vscode.TextDocument, range: vscode.Range) => {
                const code = document.getText(range);

                const prompt = `Optimize this ${document.languageId} code for performance and readability:

\`\`\`${document.languageId}
${code}
\`\`\`

Explain the optimizations made.`;

                await sendToChat(prompt);
            }
        )
    );

    // Add error handling
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.addErrorHandling',
            async (document: vscode.TextDocument, range: vscode.Range) => {
                const code = document.getText(range);

                const prompt = `Add proper error handling to this ${document.languageId} code:

\`\`\`${document.languageId}
${code}
\`\`\`

Include try-catch blocks, input validation, and meaningful error messages as appropriate.`;

                await sendToChat(prompt);
            }
        )
    );

    // Implement TODO
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.implementTodo',
            async (document: vscode.TextDocument, line: vscode.TextLine) => {
                // Extract TODO comment
                const todoMatch = line.text.match(/\b(TODO|FIXME|HACK|XXX)[:\s]*(.*)$/i);
                const todoText = todoMatch ? todoMatch[2].trim() : line.text;

                // Get surrounding context
                const contextRange = new vscode.Range(
                    Math.max(0, line.lineNumber - 10),
                    0,
                    Math.min(document.lineCount - 1, line.lineNumber + 10),
                    0
                );
                const context = document.getText(contextRange);

                const prompt = `Implement this TODO in ${document.languageId}:

TODO: ${todoText}

Context:
\`\`\`${document.languageId}
${context}
\`\`\`

Provide the implementation code.`;

                await sendToChat(prompt);
            }
        )
    );

    // Register code action provider
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            { scheme: 'file' },
            new VictorCodeActionProvider(client),
            {
                providedCodeActionKinds: VictorCodeActionProvider.providedCodeActionKinds,
            }
        )
    );
}

/**
 * Get lines of code around a range for context
 */
function getContextLines(
    document: vscode.TextDocument,
    range: vscode.Range,
    surroundingLines: number
): string {
    const startLine = Math.max(0, range.start.line - surroundingLines);
    const endLine = Math.min(document.lineCount - 1, range.end.line + surroundingLines);

    const lines: string[] = [];
    for (let i = startLine; i <= endLine; i++) {
        const lineText = document.lineAt(i).text;
        const marker = i >= range.start.line && i <= range.end.line ? '>>>' : '   ';
        lines.push(`${marker} ${i + 1}: ${lineText}`);
    }

    return lines.join('\n');
}
