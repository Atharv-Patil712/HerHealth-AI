export function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2 font-bold text-lg ${className}`}>
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-rose-600 text-white text-sm">
        H
      </span>
      <span>HerHealth AI</span>
    </div>
  );
}

export default Logo;