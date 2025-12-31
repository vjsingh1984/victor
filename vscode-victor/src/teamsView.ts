/**
 * Teams View Provider
 *
 * Provides a TreeView for managing and monitoring agent teams.
 * Displays teams, their members, status, and real-time communication.
 *
 * Features:
 * - Team creation and configuration
 * - Member status visualization
 * - Inter-agent communication log
 * - Team execution control
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

// Team formation types
export enum TeamFormation {
    Sequential = 'sequential',
    Parallel = 'parallel',
    Hierarchical = 'hierarchical',
    Pipeline = 'pipeline',
}

// Member status
export enum MemberStatus {
    Pending = 'pending',
    Running = 'running',
    Completed = 'completed',
    Failed = 'failed',
}

// Team status
export enum TeamStatus {
    Draft = 'draft',
    Running = 'running',
    Paused = 'paused',
    Completed = 'completed',
    Failed = 'failed',
    Cancelled = 'cancelled',
}

// Team member representation
export interface TeamMember {
    id: string;
    role: 'researcher' | 'planner' | 'executor' | 'reviewer' | 'tester';
    name: string;
    goal: string;
    status: MemberStatus;
    toolBudget: number;
    toolsUsed: number;
    discoveries: string[];
    isManager?: boolean;
}

// Team representation
export interface Team {
    id: string;
    name: string;
    goal: string;
    formation: TeamFormation;
    status: TeamStatus;
    members: TeamMember[];
    totalToolBudget: number;
    totalToolsUsed: number;
    startTime: number;
    endTime?: number;
    currentStep?: string;
    output?: string;
    error?: string;
}

// Inter-agent message
export interface TeamMessage {
    id: string;
    timestamp: number;
    senderId: string;
    senderRole: string;
    type: 'discovery' | 'request' | 'status' | 'handoff';
    content: string;
}

// Tree item types
type TeamTreeItem = TeamItem | TeamMemberItem | TeamInfoItem | TeamMessageItem;

/**
 * Tree item representing a team
 */
class TeamItem extends vscode.TreeItem {
    constructor(
        public readonly team: Team,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(team.name, collapsibleState);

        this.id = team.id;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `team-${team.status}`;
    }

    private _getDescription(): string {
        const memberCount = this.team.members.length;
        const formationLabel = this.team.formation.charAt(0).toUpperCase() + this.team.formation.slice(1);
        return `${formationLabel} • ${memberCount} member${memberCount !== 1 ? 's' : ''} • ${this._getStatusLabel()}`;
    }

    private _getStatusLabel(): string {
        switch (this.team.status) {
            case TeamStatus.Draft: return '📝 Draft';
            case TeamStatus.Running: return '🔄 Running';
            case TeamStatus.Paused: return '⏸️ Paused';
            case TeamStatus.Completed: return '✅ Done';
            case TeamStatus.Failed: return '❌ Failed';
            case TeamStatus.Cancelled: return '🚫 Cancelled';
            default: return this.team.status;
        }
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.team.name}\n\n`);
        md.appendMarkdown(`**Goal:** ${this.team.goal}\n\n`);
        md.appendMarkdown(`**Formation:** ${this.team.formation}\n\n`);
        md.appendMarkdown(`**Status:** ${this._getStatusLabel()}\n\n`);
        md.appendMarkdown(`**Members:**\n`);
        for (const member of this.team.members) {
            const managerBadge = member.isManager ? ' 👑' : '';
            md.appendMarkdown(`- ${member.name} (${member.role})${managerBadge}\n`);
        }
        md.appendMarkdown(`\n**Tool Budget:** ${this.team.totalToolsUsed}/${this.team.totalToolBudget}\n`);

        if (this.team.error) {
            md.appendMarkdown(`\n**Error:** ${this.team.error}\n`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.team.status) {
            case TeamStatus.Draft:
                return new vscode.ThemeIcon('edit', new vscode.ThemeColor('charts.gray'));
            case TeamStatus.Running:
                return new vscode.ThemeIcon('run-all', new vscode.ThemeColor('charts.blue'));
            case TeamStatus.Paused:
                return new vscode.ThemeIcon('debug-pause', new vscode.ThemeColor('charts.orange'));
            case TeamStatus.Completed:
                return new vscode.ThemeIcon('pass', new vscode.ThemeColor('charts.green'));
            case TeamStatus.Failed:
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('charts.red'));
            case TeamStatus.Cancelled:
                return new vscode.ThemeIcon('circle-slash', new vscode.ThemeColor('charts.gray'));
            default:
                return new vscode.ThemeIcon('organization');
        }
    }
}

/**
 * Tree item representing a team member
 */
class TeamMemberItem extends vscode.TreeItem {
    constructor(
        public readonly member: TeamMember,
        public readonly teamId: string
    ) {
        super(member.name, vscode.TreeItemCollapsibleState.None);

        this.id = `${teamId}-${member.id}`;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `member-${member.status}`;
    }

    private _getDescription(): string {
        const managerBadge = this.member.isManager ? '👑 ' : '';
        const toolUsage = `${this.member.toolsUsed}/${this.member.toolBudget} tools`;
        return `${managerBadge}${this.member.role} • ${toolUsage}`;
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.member.name}\n\n`);
        md.appendMarkdown(`**Role:** ${this.member.role}\n\n`);
        md.appendMarkdown(`**Goal:** ${this.member.goal}\n\n`);
        md.appendMarkdown(`**Status:** ${this.member.status}\n\n`);

        if (this.member.discoveries.length > 0) {
            md.appendMarkdown(`**Discoveries:**\n`);
            for (const disc of this.member.discoveries.slice(0, 5)) {
                md.appendMarkdown(`- ${disc}\n`);
            }
            if (this.member.discoveries.length > 5) {
                md.appendMarkdown(`- ... and ${this.member.discoveries.length - 5} more\n`);
            }
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        const roleIcons: Record<string, string> = {
            'researcher': 'search',
            'planner': 'list-tree',
            'executor': 'play',
            'reviewer': 'eye',
            'tester': 'beaker',
        };

        const icon = roleIcons[this.member.role] || 'person';

        switch (this.member.status) {
            case MemberStatus.Pending:
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.gray'));
            case MemberStatus.Running:
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.blue'));
            case MemberStatus.Completed:
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.green'));
            case MemberStatus.Failed:
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.red'));
            default:
                return new vscode.ThemeIcon(icon);
        }
    }
}

/**
 * Tree item for team info
 */
class TeamInfoItem extends vscode.TreeItem {
    constructor(label: string, value: string, icon?: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = value;
        this.iconPath = icon ? new vscode.ThemeIcon(icon) : undefined;
        this.contextValue = 'team-info';
    }
}

/**
 * Tree item for team messages
 */
class TeamMessageItem extends vscode.TreeItem {
    constructor(
        public readonly message: TeamMessage,
        public readonly teamId: string
    ) {
        super(`[${message.senderRole}] ${message.content.slice(0, 50)}...`, vscode.TreeItemCollapsibleState.None);

        this.id = `${teamId}-msg-${message.id}`;
        this.description = new Date(message.timestamp).toLocaleTimeString();
        this.tooltip = new vscode.MarkdownString(`**${message.type.toUpperCase()}**\n\n${message.content}`);
        this.iconPath = this._getIcon();
        this.contextValue = 'team-message';
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.message.type) {
            case 'discovery':
                return new vscode.ThemeIcon('lightbulb', new vscode.ThemeColor('charts.yellow'));
            case 'request':
                return new vscode.ThemeIcon('question', new vscode.ThemeColor('charts.blue'));
            case 'status':
                return new vscode.ThemeIcon('info', new vscode.ThemeColor('charts.gray'));
            case 'handoff':
                return new vscode.ThemeIcon('arrow-right', new vscode.ThemeColor('charts.green'));
            default:
                return new vscode.ThemeIcon('comment');
        }
    }
}

/**
 * Teams View TreeDataProvider
 */
export class TeamsViewProvider implements vscode.TreeDataProvider<TeamTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData: vscode.EventEmitter<TeamTreeItem | undefined | null | void> =
        new vscode.EventEmitter<TeamTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TeamTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private _teams: Map<string, Team> = new Map();
    private _messages: Map<string, TeamMessage[]> = new Map();
    private _disposables: vscode.Disposable[] = [];
    private _refreshInterval: NodeJS.Timeout | null = null;
    private _isLoading: boolean = false;
    private _lastFetchTime: number = 0;
    private _statusBarItem: vscode.StatusBarItem;

    constructor(
        private readonly _client: VictorClient,
        private readonly _outputChannel?: vscode.OutputChannel
    ) {
        // Create status bar item
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            97
        );
        this._statusBarItem.command = 'victor.createTeam';
        this._updateStatusBar();
        this._disposables.push(this._statusBarItem);

        // Subscribe to team events
        this._subscribeToTeamEvents();

        // Initial fetch
        this._fetchTeamsFromBackend();

        // Start auto-refresh
        this._startAutoRefresh();
    }

    private _updateStatusBar(): void {
        const activeCount = this.getActiveCount();
        if (activeCount > 0) {
            this._statusBarItem.text = `$(organization) ${activeCount} team${activeCount > 1 ? 's' : ''}`;
            this._statusBarItem.tooltip = `${activeCount} active team${activeCount > 1 ? 's' : ''}. Click to create new team.`;
            this._statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.prominentBackground');
            this._statusBarItem.show();
        } else {
            this._statusBarItem.text = '$(organization) Teams';
            this._statusBarItem.tooltip = 'Create a new agent team';
            this._statusBarItem.backgroundColor = undefined;
            this._statusBarItem.show();
        }
    }

    private _subscribeToTeamEvents(): void {
        const eventHandler = (event: { type: string; data: unknown }) => {
            switch (event.type) {
                case 'team_created':
                    this._handleTeamCreated(event.data);
                    break;
                case 'team_started':
                case 'team_updated':
                    this._handleTeamUpdate(event.data);
                    break;
                case 'team_member_update':
                    this._handleMemberUpdate(event.data);
                    break;
                case 'team_message':
                    this._handleTeamMessage(event.data);
                    break;
                case 'team_completed':
                case 'team_failed':
                case 'team_cancelled':
                    this._handleTeamUpdate(event.data);
                    break;
            }
        };

        // Register for team events (similar pattern to agent events)
        this._client.onAgentEvent((event) => {
            if (event.type.startsWith('team_')) {
                eventHandler(event);
            }
        });

        this._log('Subscribed to team events');
    }

    private _handleTeamCreated(data: unknown): void {
        const team = this._convertBackendTeam(data);
        this._teams.set(team.id, team);
        this._messages.set(team.id, []);
        this.refresh();
        this._updateStatusBar();
        this._log(`Team created: ${team.id} - ${team.name}`);
    }

    private _handleTeamUpdate(data: unknown): void {
        const team = this._convertBackendTeam(data);
        this._teams.set(team.id, team);
        this.refresh();
        this._updateStatusBar();
        this._log(`Team updated: ${team.id} - ${team.status}`);
    }

    private _handleMemberUpdate(data: unknown): void {
        const memberData = data as { team_id: string; member: TeamMember };
        const team = this._teams.get(memberData.team_id);
        if (team) {
            const memberIndex = team.members.findIndex(m => m.id === memberData.member.id);
            if (memberIndex >= 0) {
                team.members[memberIndex] = memberData.member;
                this.refresh();
            }
        }
    }

    private _handleTeamMessage(data: unknown): void {
        const msgData = data as { team_id: string; message: TeamMessage };
        const messages = this._messages.get(msgData.team_id) || [];
        messages.push(msgData.message);
        this._messages.set(msgData.team_id, messages);
        this.refresh();
    }

    private _convertBackendTeam(data: unknown): Team {
        const d = data as Record<string, unknown>;
        return {
            id: d.id as string,
            name: d.name as string || 'Unnamed Team',
            goal: d.goal as string || '',
            formation: d.formation as TeamFormation || TeamFormation.Sequential,
            status: d.status as TeamStatus || TeamStatus.Draft,
            members: (d.members as TeamMember[]) || [],
            totalToolBudget: d.total_tool_budget as number || 0,
            totalToolsUsed: d.total_tools_used as number || 0,
            startTime: (d.start_time as number) * 1000 || Date.now(),
            endTime: d.end_time ? (d.end_time as number) * 1000 : undefined,
            currentStep: d.current_step as string | undefined,
            output: d.output as string | undefined,
            error: d.error as string | undefined,
        };
    }

    async _fetchTeamsFromBackend(): Promise<void> {
        const now = Date.now();
        if (now - this._lastFetchTime < 1000) {
            return;
        }
        this._lastFetchTime = now;

        if (this._isLoading) { return; }
        this._isLoading = true;

        try {
            const teams = await this._client.listTeams();

            this._teams.clear();
            for (const teamData of teams) {
                const team = this._convertBackendTeam(teamData);
                this._teams.set(team.id, team);
            }

            this.refresh();
            this._updateStatusBar();
            this._log(`Fetched ${teams.length} teams from backend`);
        } catch (error) {
            this._log(`Failed to fetch teams: ${error}`);
        } finally {
            this._isLoading = false;
        }
    }

    dispose(): void {
        if (this._refreshInterval) {
            clearInterval(this._refreshInterval);
        }
        this._disposables.forEach(d => d.dispose());
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    // TreeDataProvider implementation
    getTreeItem(element: TeamTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: TeamTreeItem): Promise<TeamTreeItem[]> {
        if (!element) {
            return this._getTeamItems();
        }

        if (element instanceof TeamItem) {
            return this._getTeamChildren(element.team);
        }

        return [];
    }

    getParent(element: TeamTreeItem): vscode.ProviderResult<TeamTreeItem> {
        if (element instanceof TeamMemberItem) {
            const team = this._teams.get(element.teamId);
            if (team) {
                return new TeamItem(team, vscode.TreeItemCollapsibleState.Expanded);
            }
        }
        return null;
    }

    // Team management methods
    async createTeam(name: string, goal: string, formation: TeamFormation, memberRoles: string[]): Promise<string | null> {
        try {
            const teamId = await this._client.createTeam({
                name,
                goal,
                formation,
                members: memberRoles.map((role, i) => ({
                    id: `${role}_${i + 1}`,
                    role,
                    name: `${role.charAt(0).toUpperCase()}${role.slice(1)} ${i + 1}`,
                    goal: `Perform ${role} tasks`,
                })),
            });
            this._log(`Created team: ${teamId}`);
            return teamId;
        } catch (error) {
            this._log(`Failed to create team: ${error}`);
            vscode.window.showErrorMessage(`Failed to create team: ${error}`);
            return null;
        }
    }

    async startTeam(teamId: string): Promise<void> {
        try {
            await this._client.startTeam(teamId);
            this._log(`Started team: ${teamId}`);
        } catch (error) {
            this._log(`Failed to start team: ${error}`);
            vscode.window.showErrorMessage(`Failed to start team: ${error}`);
        }
    }

    async cancelTeam(teamId: string): Promise<void> {
        try {
            await this._client.cancelTeam(teamId);
            this._log(`Cancelled team: ${teamId}`);
        } catch (error) {
            this._log(`Failed to cancel team: ${error}`);
            vscode.window.showErrorMessage(`Failed to cancel team: ${error}`);
        }
    }

    async clearCompleted(): Promise<void> {
        try {
            await this._client.clearTeams();

            for (const [id, team] of this._teams.entries()) {
                if (team.status === TeamStatus.Completed ||
                    team.status === TeamStatus.Failed ||
                    team.status === TeamStatus.Cancelled) {
                    this._teams.delete(id);
                    this._messages.delete(id);
                }
            }
            this.refresh();
            this._log('Cleared completed teams');
        } catch (error) {
            this._log(`Failed to clear teams: ${error}`);
        }
    }

    getActiveCount(): number {
        let count = 0;
        for (const team of this._teams.values()) {
            if (team.status === TeamStatus.Running) {
                count++;
            }
        }
        return count;
    }

    private _getTeamItems(): TeamTreeItem[] {
        if (this._teams.size === 0) {
            return [new TeamInfoItem('No teams', 'Create a team to get started', 'info')];
        }

        const items: TeamItem[] = [];

        // Sort: running first, then by start time
        const sorted = Array.from(this._teams.values()).sort((a, b) => {
            const statusOrder = {
                [TeamStatus.Running]: 0,
                [TeamStatus.Draft]: 1,
                [TeamStatus.Paused]: 2,
                [TeamStatus.Completed]: 3,
                [TeamStatus.Failed]: 4,
                [TeamStatus.Cancelled]: 5,
            };
            const statusDiff = statusOrder[a.status] - statusOrder[b.status];
            if (statusDiff !== 0) { return statusDiff; }
            return b.startTime - a.startTime;
        });

        for (const team of sorted) {
            const hasChildren = team.members.length > 0;
            items.push(new TeamItem(
                team,
                hasChildren
                    ? vscode.TreeItemCollapsibleState.Collapsed
                    : vscode.TreeItemCollapsibleState.None
            ));
        }

        return items;
    }

    private _getTeamChildren(team: Team): TeamTreeItem[] {
        const children: TeamTreeItem[] = [];

        // Add members
        for (const member of team.members) {
            children.push(new TeamMemberItem(member, team.id));
        }

        // Add recent messages (last 5)
        const messages = this._messages.get(team.id) || [];
        if (messages.length > 0) {
            children.push(new TeamInfoItem('Recent Messages', `${messages.length} messages`, 'comment-discussion'));
            for (const msg of messages.slice(-5)) {
                children.push(new TeamMessageItem(msg, team.id));
            }
        }

        return children;
    }

    private _startAutoRefresh(): void {
        this._refreshInterval = setInterval(() => {
            const hasRunning = Array.from(this._teams.values()).some(
                t => t.status === TeamStatus.Running
            );
            if (hasRunning) {
                this.refresh();
            }
        }, 2000);
    }

    private _log(message: string): void {
        this._outputChannel?.appendLine(`[Teams] ${message}`);
    }
}

/**
 * Register team-related commands
 */
export function registerTeamCommands(
    context: vscode.ExtensionContext,
    teamsView: TeamsViewProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshTeams', async () => {
            await teamsView._fetchTeamsFromBackend();
        }),

        vscode.commands.registerCommand('victor.clearTeams', async () => {
            await teamsView.clearCompleted();
            vscode.window.showInformationMessage('Cleared completed teams');
        }),

        vscode.commands.registerCommand('victor.createTeam', async () => {
            // Get team name
            const name = await vscode.window.showInputBox({
                prompt: 'Enter team name',
                placeHolder: 'e.g., Code Review Team',
                validateInput: (value) => {
                    if (!value || value.trim().length < 2) {
                        return 'Please enter a valid team name';
                    }
                    return null;
                }
            });
            if (!name) { return; }

            // Get team goal
            const goal = await vscode.window.showInputBox({
                prompt: 'Enter team goal',
                placeHolder: 'e.g., Review authentication code for security issues',
                validateInput: (value) => {
                    if (!value || value.trim().length < 5) {
                        return 'Please enter a valid goal (at least 5 characters)';
                    }
                    return null;
                }
            });
            if (!goal) { return; }

            // Select formation
            const formations = [
                { label: 'Sequential', description: 'Members execute one after another', value: TeamFormation.Sequential },
                { label: 'Parallel', description: 'All members execute simultaneously', value: TeamFormation.Parallel },
                { label: 'Hierarchical', description: 'Manager delegates to members', value: TeamFormation.Hierarchical },
                { label: 'Pipeline', description: 'Output of one feeds to next', value: TeamFormation.Pipeline },
            ];

            const formationChoice = await vscode.window.showQuickPick(formations, {
                placeHolder: 'Select team formation',
            });
            if (!formationChoice) { return; }

            // Select member roles
            const roleOptions = [
                { label: 'Researcher', description: 'Search and analyze code', picked: true },
                { label: 'Planner', description: 'Create implementation plans', picked: false },
                { label: 'Executor', description: 'Make code changes', picked: true },
                { label: 'Reviewer', description: 'Review and validate changes', picked: false },
                { label: 'Tester', description: 'Write and run tests', picked: false },
            ];

            const roleChoices = await vscode.window.showQuickPick(roleOptions, {
                placeHolder: 'Select team member roles',
                canPickMany: true,
            });
            if (!roleChoices || roleChoices.length === 0) { return; }

            const memberRoles = roleChoices.map(r => r.label.toLowerCase());

            // Create the team
            const teamId = await teamsView.createTeam(name, goal, formationChoice.value, memberRoles);
            if (teamId) {
                vscode.window.showInformationMessage(`Created team: ${name}`);

                // Ask if user wants to start immediately
                const start = await vscode.window.showQuickPick([
                    { label: 'Start Now', value: true },
                    { label: 'Start Later', value: false },
                ], {
                    placeHolder: 'Start team execution?',
                });

                if (start?.value) {
                    await teamsView.startTeam(teamId);
                }
            }
        }),

        vscode.commands.registerCommand('victor.startTeam', async (item: TeamItem) => {
            if (item?.team?.id) {
                await teamsView.startTeam(item.team.id);
                vscode.window.showInformationMessage(`Started team: ${item.team.name}`);
            }
        }),

        vscode.commands.registerCommand('victor.cancelTeam', async (item: TeamItem) => {
            if (item?.team?.id) {
                await teamsView.cancelTeam(item.team.id);
                vscode.window.showInformationMessage(`Cancelled team: ${item.team.name}`);
            }
        }),

        vscode.commands.registerCommand('victor.viewTeamOutput', async (item: TeamItem) => {
            if (!item?.team) { return; }

            const team = item.team;
            const content = [
                `# Team: ${team.name}`,
                `**Status:** ${team.status}`,
                `**Formation:** ${team.formation}`,
                `**Goal:** ${team.goal}`,
                '',
                '## Members',
            ];

            for (const member of team.members) {
                const managerBadge = member.isManager ? ' 👑' : '';
                content.push(`- **${member.name}** (${member.role})${managerBadge} - ${member.status}`);
                if (member.discoveries.length > 0) {
                    content.push('  Discoveries:');
                    for (const disc of member.discoveries) {
                        content.push(`  - ${disc}`);
                    }
                }
            }

            if (team.output) {
                content.push('', '## Output', '```', team.output, '```');
            }

            if (team.error) {
                content.push('', '## Error', '```', team.error, '```');
            }

            const doc = await vscode.workspace.openTextDocument({
                content: content.join('\n'),
                language: 'markdown',
            });
            await vscode.window.showTextDocument(doc, { preview: true });
        })
    );
}
