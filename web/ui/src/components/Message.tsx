import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import mermaid from 'mermaid';
import Asciidoctor from '@asciidoctor/core';
import 'katex/dist/katex.min.css';

const asciidoctor = Asciidoctor();

type Sender = 'user' | 'assistant';
type Kind = 'normal' | 'tool';

interface MessageProps {
  text: string;
  sender: Sender;
  kind?: Kind;
  streaming?: boolean;
}

let mermaidId = 0;

function MermaidDiagram({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const id = useMemo(() => `mermaid-${mermaidId++}`, []);

  useEffect(() => {
    if (!ref.current) return;
    try {
      mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
      mermaid.render(id, code, (svgCode: string) => {
        if (ref.current) {
          ref.current.innerHTML = svgCode;
        }
      });
    } catch (err) {
      if (ref.current) {
        ref.current.innerText = `Mermaid render error: ${String(err)}`;
      }
    }
  }, [code, id]);

  return <div ref={ref} className="overflow-auto p-3 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700" />;
}

type ViewMode = 'render' | 'raw';

function CodeBlock({ className, children, viewMode }: { className?: string; children: React.ReactNode; viewMode: ViewMode }) {
  const language = (className || '').replace('language-', '').trim();

  const [copied, setCopied] = useState(false);
  const [plantSvg, setPlantSvg] = useState<string | null>(null);
  const [plantError, setPlantError] = useState<string | null>(null);

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Copy failed', err);
    }
  };

  const rawBlock = (
    <div className="relative">
      <button
        onClick={() => handleCopy(String(children))}
        className="absolute top-1 right-1 text-xs px-2 py-1 rounded bg-gray-700 text-white hover:bg-gray-600"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="bg-gray-900 text-gray-100 text-sm rounded p-3 overflow-auto pr-12">
        <code>{children}</code>
      </pre>
    </div>
  );

  if (viewMode === 'raw') {
    return rawBlock;
  }

  if (language === 'mermaid' || language === 'diagram') {
    return <MermaidDiagram code={String(children)} />;
  }

  if (language === 'plantuml') {
    useEffect(() => {
      let aborted = false;
      setPlantError(null);
      setPlantSvg(null);
      fetch('/render/plantuml', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: String(children),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          const text = await res.text();
          if (!aborted) setPlantSvg(text);
        })
        .catch((err) => {
          if (!aborted) setPlantError(String(err));
        });
      return () => {
        aborted = true;
      };
    }, [children]);

    if (plantError) {
      return (
        <div className="bg-red-50 text-red-700 border border-red-200 rounded p-2 text-sm">
          PlantUML render error: {plantError}
        </div>
      );
    }
    if (!plantSvg) {
      return <div className="text-sm text-gray-500">Rendering PlantUML…</div>;
    }
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: plantSvg }} />;
  }

  if (language === 'drawio' || language === 'lucid' || language === 'drawio-xml') {
    const [diagramSvg, setDiagramSvg] = useState<string | null>(null);
    const [diagramError, setDiagramError] = useState<string | null>(null);
    useEffect(() => {
      let aborted = false;
      setDiagramError(null);
      setDiagramSvg(null);
      fetch('/render/drawio', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: String(children),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          const text = await res.text();
          if (!aborted) setDiagramSvg(text);
        })
        .catch((err) => {
          if (!aborted) setDiagramError(String(err));
        });
      return () => {
        aborted = true;
      };
    }, [children]);

    if (diagramError) {
      return (
        <div className="bg-red-50 text-red-700 border border-red-200 rounded p-2 text-sm">
          Draw.io render error: {diagramError}
        </div>
      );
    }
    if (!diagramSvg) {
      return <div className="text-sm text-gray-500">Rendering Draw.io…</div>;
    }
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: diagramSvg }} />;
  }

  if (language === 'graphviz' || language === 'dot') {
    const [graphvizSvg, setGraphvizSvg] = useState<string | null>(null);
    const [graphvizError, setGraphvizError] = useState<string | null>(null);
    useEffect(() => {
      let aborted = false;
      setGraphvizError(null);
      setGraphvizSvg(null);
      fetch('/render/graphviz', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: String(children),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          const text = await res.text();
          if (!aborted) setGraphvizSvg(text);
        })
        .catch((err) => {
          if (!aborted) setGraphvizError(String(err));
        });
      return () => {
        aborted = true;
      };
    }, [children]);

    if (graphvizError) {
      return (
        <div className="bg-red-50 text-red-700 border border-red-200 rounded p-2 text-sm">
          Graphviz render error: {graphvizError}
        </div>
      );
    }
    if (!graphvizSvg) {
      return <div className="text-sm text-gray-500">Rendering Graphviz…</div>;
    }
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: graphvizSvg }} />;
  }

  if (language === 'd2') {
    const [d2Svg, setD2Svg] = useState<string | null>(null);
    const [d2Error, setD2Error] = useState<string | null>(null);
    useEffect(() => {
      let aborted = false;
      setD2Error(null);
      setD2Svg(null);
      fetch('/render/d2', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: String(children),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          const text = await res.text();
          if (!aborted) setD2Svg(text);
        })
        .catch((err) => {
          if (!aborted) setD2Error(String(err));
        });
      return () => {
        aborted = true;
      };
    }, [children]);

    if (d2Error) {
      return (
        <div className="bg-red-50 text-red-700 border border-red-200 rounded p-2 text-sm">
          D2 render error: {d2Error}
        </div>
      );
    }
    if (!d2Svg) {
      return <div className="text-sm text-gray-500">Rendering D2…</div>;
    }
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: d2Svg }} />;
  }

  if (language === 'asciidoc' || language === 'adoc') {
    try {
      const html = asciidoctor.convert(String(children)) as string;
      return (
        <div className="bg-white dark:bg-gray-800 text-sm rounded border border-gray-200 dark:border-gray-700 p-3 overflow-auto prose dark:prose-invert"
             dangerouslySetInnerHTML={{ __html: html }} />
      );
    } catch (err) {
      return (
        <div className="bg-red-50 text-red-700 border border-red-200 rounded p-2 text-sm">
          AsciiDoc render error: {String(err)}
        </div>
      );
    }
  }

  return rawBlock;
}

function Message({ text, sender, kind = 'normal', streaming = false }: MessageProps) {
  const isUser = sender === 'user';
  const isTool = kind === 'tool';
  const [viewMode, setViewMode] = useState<ViewMode>(streaming ? 'raw' : 'render');
  const [copiedFull, setCopiedFull] = useState(false);

  const bubbleClasses = isUser
    ? 'bg-blue-600 text-white rounded-br-none'
    : isTool
      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100 rounded-bl-none border border-yellow-200 dark:border-yellow-800'
      : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-none';

  return (
    <div className={`flex items-start ${isUser ? 'justify-end' : 'justify-start'} w-full`}>
      <div className={`relative w-full md:w-11/12 lg:w-3/4 max-w-3xl px-4 py-3 rounded-2xl shadow-md whitespace-pre-wrap ${bubbleClasses}`}>
        {!isUser && !isTool && !streaming && (
          <div className="absolute top-2 right-2 flex items-center gap-2 text-xs">
            <button
              className="px-2 py-1 rounded bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-white"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(text);
                  setCopiedFull(true);
                  setTimeout(() => setCopiedFull(false), 1200);
                } catch (err) {
                  console.error('Copy full failed', err);
                }
              }}
            >
              {copiedFull ? 'Copied' : 'Copy full'}
            </button>
            <span className="text-gray-600 dark:text-gray-300">View:</span>
            <button
              className={`px-2 py-1 rounded ${viewMode === 'render' ? 'bg-blue-600 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-white'}`}
              onClick={() => setViewMode('render')}
            >
              Render
            </button>
            <button
              className={`px-2 py-1 rounded ${viewMode === 'raw' ? 'bg-blue-600 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-white'}`}
              onClick={() => setViewMode('raw')}
            >
              Raw
            </button>
          </div>
        )}
        {isTool ? (
          <p className="text-xs font-mono">{text.replace(/^\[tool\]\s*/i, '')}</p>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              code: ({ className, children }) => <CodeBlock className={className} viewMode={viewMode}>{children}</CodeBlock>,
            }}
          >
            {text}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}

export default Message;
