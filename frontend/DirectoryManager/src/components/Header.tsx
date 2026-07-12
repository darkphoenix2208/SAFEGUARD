import { ArrowLeft, Home, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";

const Header = () => {
  return (
    <header className="relative z-10 flex items-center justify-between p-6 bg-background/80 backdrop-blur-sm border-b border-border/50">
      {/* Left side - Navigation buttons */}
      <div className="flex items-center gap-3">
        <Button 
          variant="ghost" 
          size="sm" 
          className="nav-button"
          onClick={() => window.location.href = 'http://localhost:3000'}
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
        <Button 
          variant="ghost" 
          size="sm" 
          className="nav-button"
          onClick={() => window.location.href = 'http://localhost:3000'}
        >
          <Home className="w-4 h-4 mr-2" />
          Home
        </Button>
      </div>

      {/* Right side - Logo */}
      <div className="flex items-center gap-2">
        <div className="p-2 rounded-lg bg-gradient-to-br from-primary to-primary-glow shadow-lg">
          <Shield className="w-6 h-6 text-white" />
        </div>
        <span className="text-lg font-semibold text-foreground">Safeguard</span>
      </div>
    </header>
  );
};

export default Header;