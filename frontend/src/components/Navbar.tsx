import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="fixed top-0 w-full z-50 bg-bg/80 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="font-display font-bold text-xl text-text flex items-center gap-2">
              <span className="text-accent text-2xl">⚡</span> KinetiCore
            </Link>
          </div>
          <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
            <Link href="/dashboard" className="text-muted hover:text-accent transition-colors px-3 py-2 text-sm font-medium">Dashboard</Link>
            <Link href="/select-exercise" className="text-muted hover:text-accent transition-colors px-3 py-2 text-sm font-medium">Exercises</Link>
            <Link href="/results" className="text-muted hover:text-accent transition-colors px-3 py-2 text-sm font-medium">Results</Link>
          </div>
          <div className="flex items-center">
            <Link href="/login" className="bg-surface2 border border-border text-text hover:border-accent transition-colors px-4 py-2 rounded-full text-sm font-bold">
              Sign In
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
