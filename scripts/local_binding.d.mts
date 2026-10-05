export function activeLocalConfigPath(): Promise<string | null>;
export function assertActiveMemoryConnection(config: { baseUrl: string; apiKey?: string }): Promise<void>;
export function assertCloudProvidersAllowed(): Promise<void>;
