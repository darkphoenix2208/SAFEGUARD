import { cn } from "@/lib/utils";

interface SecurityCardProps {
  title: string;
  backgroundImage: string;
  className?: string;
  onClick?: () => void;
}

const SecurityCard = ({ title, backgroundImage, className, onClick }: SecurityCardProps) => {
  return (
    <div
      className={cn(
        "tech-card group cursor-pointer p-8 min-h-[280px] flex items-end",
        className
      )}
      onClick={onClick}
      style={{
        backgroundImage: `linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(94,8,2,0.3) 100%), url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      <div className="relative z-10 w-full">
        <h3 className="text-2xl font-semibold text-white mb-2 group-hover:text-primary-glow transition-colors duration-300">
          {title}
        </h3>
        <div className="w-12 h-1 bg-primary group-hover:bg-primary-glow transition-all duration-300 group-hover:w-20"></div>
      </div>
      
      {/* Overlay effect */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-60 group-hover:opacity-40 transition-opacity duration-300 rounded-xl"></div>
    </div>
  );
};

export default SecurityCard;