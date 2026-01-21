Name:           victor-ai
Version:        1.0.0
Release:        1%{?dist}
Summary:        Open-source AI coding assistant with multi-provider support

License:        Apache-2.0
URL:            https://github.com/vijayksingh/victor
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
BuildRequires:  python3-pyproject.toml

Requires:       python3-pydantic >= 2.0
Requires:       python3-typer >= 0.12
Requires:       python3-rich >= 13.7
Requires:       python3-httpx >= 0.27

Recommends:     python3-anthropic >= 0.34
Recommends:     python3-openai >= 1.40

%description
Victor AI is an open-source AI coding assistant supporting 21 LLM
providers with 55+ specialized tools across 5 domain verticals
(Coding, DevOps, RAG, Data Analysis, Research).

Key features:
- Multi-provider support (Anthropic, OpenAI, Google, Ollama, etc.)
- Workflow orchestration with StateGraph DSL
- Semantic codebase search
- Multi-agent coordination
- Enterprise-grade security
- Comprehensive testing (92%+ pass rate)
- Extensible tool system
- CLI/TUI interfaces

%prep
%autosetup -p1

%build
%py3_build

%install
%py3_install
# Install man pages if they exist
if [ -d docs/man ]; then
    install -d %{buildroot}%{_mandir}/man1
    install -p -m 644 docs/man/*.1 %{buildroot}%{_mandir}/man1/
fi

%check
# Skip tests during RPM build (run during CI instead)

%files
%doc README.md
%doc CHANGELOG.md
%doc RELEASE_NOTES.md
%license LICENSE
%{_bindir}/victor
%{_bindir}/vic
%{python3_sitelib}/victor*
%{python3_sitelib}/victor_ai-%{version}.dist-info/

%changelog
* Tue Jan 21 2025 Vijaykumar Singh <singhvjd@gmail.com> - 1.0.0-1
- Initial release of Victor AI 1.0.0
- Production-ready AI coding assistant
- 21 LLM provider support
- 55+ specialized tools across 5 verticals
- SOLID architecture with 98 protocols
- Comprehensive security suite (132 tests)
- 72.8% faster startup with lazy loading
- Full backward compatibility with 0.5.x
