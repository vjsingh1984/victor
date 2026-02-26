@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=Victor

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:live
	sphinx-autobuild %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
	goto end

:html
	%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
	if errorlevel 1 exit /b 1
	echo.
	echo.Build finished. The HTML pages are in %BUILDDIR%/html.
	goto end

:apidoc
	@echo Generating API documentation...
	sphinx-apidoc -o api ../victor --force --module-first
	@echo API documentation generated
	goto end

:clean
	rmdir /S /Q %BUILDDIR%
	@echo Cleaned build directory
	goto end

:end
popd
