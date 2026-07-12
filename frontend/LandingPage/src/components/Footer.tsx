import { Shield, Mail, Phone, MapPin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-card border-t border-steel-gray/30 py-16">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <Shield className="h-6 w-6 text-primary" />
              <span className="text-xl font-bold bg-gradient-metallic bg-clip-text text-transparent">
                Safeguard
              </span>
            </div>
            <p className="text-muted-foreground">
              Advanced AI-powered CCTV surveillance system for modern security needs.
            </p>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Features</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li>Object Detection</li>
              <li>Motion Analysis</li>
              <li>Real-time Tracking</li>
              <li>Pose Estimation</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Company</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li>About Us</li>
              <li>Privacy Policy</li>
              <li>Terms of Service</li>
              <li>Support</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Contact</h4>
            <div className="space-y-3 text-muted-foreground">
              <div className="flex items-center space-x-2">
                <Mail className="h-4 w-4" />
                <span>info@securevista.ai</span>
              </div>
              <div className="flex items-center space-x-2">
                <Phone className="h-4 w-4" />
                <span>+1 (555) 123-4567</span>
              </div>
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4" />
                <span>San Francisco, CA</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="border-t border-steel-gray/30 mt-12 pt-8 text-center text-muted-foreground">
          <p>&copy; 2024 Safeguard. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;