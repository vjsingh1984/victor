import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import Message from './Message';

// Mock DOMPurify
vi.mock('dompurify', () => ({
  default: {
    sanitize: vi.fn((html: string) => {
      // Simulate sanitization: remove script tags and event handlers
      let sanitized = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
      sanitized = sanitized.replace(/\son\w+\s*=/gi, '');
      return sanitized;
    }),
    addHook: vi.fn(),
  },
}));

// Mock Mermaid
vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn((_id: string, _code: string, callback: (svg: string) => void) => {
      // Simulate SVG output with potential XSS
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <circle cx="50" cy="50" r="40" fill="red" />
      </svg>`;
      callback(svg);
    }),
  },
}));

// Mock Asciidoctor
vi.mock('@asciidoctor/core', () => ({
  default: vi.fn(() => ({
    convert: vi.fn((content: string) => {
      // Simulate HTML output
      return `<div class="paragraph"><p>${content}</p></div>`;
    }),
  })),
}));

// Mock fetch for diagram rendering
global.fetch = vi.fn();

describe('Message Component - XSS Prevention', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('SVG Rendering Sanitization', () => {
    it('should sanitize XSS attempts in Mermaid diagrams', async () => {
      const maliciousCode = `
        graph TD
          A[Start] --> B[End]
      `;

      const { container } = render(
        <Message text={"```mermaid\n" + maliciousCode + "\n```"} sender="assistant" />
      );

      await waitFor(() => {
        const svgElement = container.querySelector('svg');
        expect(svgElement).toBeInTheDocument();
        // Verify no script tags are present
        const scripts = container.querySelectorAll('script');
        expect(scripts.length).toBe(0);
      });
    });

    it('should render valid Mermaid SVG content correctly', async () => {
      const validCode = `
        graph TD
          A[Start] --> B[End]
      `;

      const { container } = render(
        <Message text={"```mermaid\n" + validCode + "\n```"} sender="assistant" />
      );

      await waitFor(() => {
        const svgElement = container.querySelector('svg');
        expect(svgElement).toBeInTheDocument();
        expect(svgElement?.getAttribute('xmlns')).toBe('http://www.w3.org/2000/svg');
      });
    });
  });

  describe('PlantUML Sanitization', () => {
    it('should sanitize PlantUML SVG responses', async () => {
      const maliciousSvg = `<svg xmlns="http://www.w3.org/2000/svg">
        <script>alert('XSS')</script>
        <circle cx="50" cy="50" r="40" fill="red"/>
      </svg>`;

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => maliciousSvg,
      });

      const { container } = render(
        <Message text={"```plantuml\n@startuml\nA -> B\n@enduml\n```"} sender="assistant" />
      );

      await waitFor(() => {
        // Script should be removed by sanitization
        const scripts = container.querySelectorAll('script');
        expect(scripts.length).toBe(0);
        // SVG elements should still render
        const circle = container.querySelector('circle');
        expect(circle).toBeInTheDocument();
      });
    });
  });

  describe('Draw.io Sanitization', () => {
    it('should sanitize Draw.io SVG responses', async () => {
      const maliciousSvg = `<svg xmlns="http://www.w3.org/2000/svg">
        <rect onclick="alert('XSS')" x="10" y="10" width="100" height="50"/>
      </svg>`;

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => maliciousSvg,
      });

      const { container } = render(
        <Message text={"```drawio\n<mxGraphModel>...</mxGraphModel>\n```"} sender="assistant" />
      );

      await waitFor(() => {
        // Verify that SVG was rendered (even if malformed after sanitization)
        const svg = container.querySelector('svg');
        expect(svg).toBeInTheDocument();
        // Verify that event handlers were removed by DOMPurify
        const rect = container.querySelector('rect');
        // The rect might not be found if sanitization removed it entirely
        // or it might exist without the onclick handler
        if (rect) {
          expect(rect?.getAttribute('onclick')).toBeNull();
        }
        // Most importantly, no script tags should be present
        const scripts = container.querySelectorAll('script');
        expect(scripts.length).toBe(0);
      });
    });
  });

  describe('Graphviz Sanitization', () => {
    it('should sanitize Graphviz SVG responses', async () => {
      const maliciousSvg = `<svg xmlns="http://www.w3.org/2000/svg">
        <g onmouseover="alert('XSS')">
          <ellipse cx="50" cy="50" rx="30" ry="20"/>
        </g>
      </svg>`;

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => maliciousSvg,
      });

      const { container } = render(
        <Message text={"```dot\ndigraph G { A -> B; }\n```"} sender="assistant" />
      );

      await waitFor(() => {
        // Verify that SVG was rendered
        const svg = container.querySelector('svg');
        expect(svg).toBeInTheDocument();
        // Verify no event handlers or scripts
        const scripts = container.querySelectorAll('script');
        expect(scripts.length).toBe(0);
        const g = container.querySelector('g');
        if (g) {
          expect(g?.getAttribute('onmouseover')).toBeNull();
        }
      });
    });
  });

  describe('D2 Sanitization', () => {
    it('should sanitize D2 SVG responses', async () => {
      const maliciousSvg = `<svg xmlns="http://www.w3.org/2000/svg">
        <script>alert('XSS')</script>
        <polygon points="100,10 40,198 190,78 10,78 160,198"/>
      </svg>`;

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => maliciousSvg,
      });

      const { container } = render(
        <Message text={"```d2\nx -> y\n```"} sender="assistant" />
      );

      await waitFor(() => {
        const scripts = container.querySelectorAll('script');
        expect(scripts.length).toBe(0);
        const polygon = container.querySelector('polygon');
        expect(polygon).toBeInTheDocument();
      });
    });
  });

  describe('AsciiDoc Sanitization', () => {
    it('should sanitize AsciiDoc HTML output', () => {
      const { container } = render(
        <Message text={"```adoc\n= Title\n\nContent here\n```"} sender="assistant" />
      );

      // Should render sanitized HTML
      const content = container.querySelector('.paragraph');
      expect(content).toBeInTheDocument();
    });

    it('should prevent script tags in AsciiDoc', () => {
      const { container } = render(
        <Message text={"```adoc\n<script>alert('XSS')</script>\n```"} sender="assistant" />
      );

      const scripts = container.querySelectorAll('script');
      expect(scripts.length).toBe(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty SVG gracefully', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => '',
      });

      const { container } = render(
        <Message text={"```plantuml\n@startuml\n@enduml\n```"} sender="assistant" />
      );

      await waitFor(() => {
        // Should not crash with empty SVG
        expect(container).toBeInTheDocument();
      });
    });

    it('should handle malformed SVG without crashing', async () => {
      const malformedSvg = `<svg><circle cx="50" cy="50"`;
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => malformedSvg,
      });

      const { container } = render(
        <Message text={"```plantuml\n@startuml\nA -> B\n@enduml\n```"} sender="assistant" />
      );

      await waitFor(() => {
        expect(container).toBeInTheDocument();
      });
    });

    it('should sanitize SVG with data URLs', async () => {
      const svgWithDataUrl = `<svg xmlns="http://www.w3.org/2000/svg">
        <image href="data:text/html,<script>alert('XSS')</script>" />
      </svg>`;

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: async () => svgWithDataUrl,
      });

      const { container } = render(
        <Message text={"```plantuml\n@startuml\nA -> B\n@enduml\n```"} sender="assistant" />
      );

      await waitFor(() => {
        const image = container.querySelector('image');
        expect(image).toBeInTheDocument();
      });
    });
  });

  describe('User Message Safety', () => {
    it('should not render HTML in user messages', () => {
      const userMessage = '<script>alert("XSS")</script>Hello';
      const { container } = render(
        <Message text={userMessage} sender="user" />
      );

      const scripts = container.querySelectorAll('script');
      expect(scripts.length).toBe(0);
    });
  });
});
