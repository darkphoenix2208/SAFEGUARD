import { Button } from "@/components/ui/button";
import { Shield, Play, Zap, Eye } from "lucide-react";

const Hero = () => {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-hero relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-32 h-32 bg-primary/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute top-40 right-20 w-48 h-48 bg-metallic/5 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute bottom-32 left-32 w-24 h-24 bg-primary/15 rounded-full blur-2xl animate-float" style={{ animationDelay: '4s' }}></div>
      </div>
      
      <div className="container mx-auto px-6 py-32 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2 text-primary">
                <Shield className="h-6 w-6" />
                <span className="text-sm font-medium uppercase tracking-wider">AI-Powered Security</span>
              </div>
              
              <h1 className="text-5xl lg:text-7xl font-bold leading-tight">
                <span className="text-primary drop-shadow-[0_0_15px_rgba(0,191,255,0.6)]">
                  Safe
                </span>
                <span className="text-foreground">guard</span>
              </h1>
              
              <h2 className="text-2xl lg:text-3xl text-muted-foreground font-light">
                Next-Generation Construction Site Safety AI
              </h2>
              
              <p className="text-lg text-muted-foreground leading-relaxed max-w-xl">
                Transform your construction site safety with Physical AI. 
                Monitor heavy machinery, track danger zone intrusions, and ensure worker 
                safety with intelligent monitoring that never sleeps.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                variant="hero" 
                size="lg" 
                className="text-lg px-8 py-6 group relative overflow-hidden"
                onClick={() => window.location.href = 'http://localhost:3001'}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                <Zap className="h-5 w-5 mr-2 group-hover:animate-bounce" />
                Start Now
              </Button>
              <Button 
                variant="metallic" 
                size="lg" 
                className="text-lg px-8 py-6 group relative overflow-hidden"
                onClick={() => window.location.href = 'http://localhost:3001'}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                <Play className="h-5 w-5 mr-2 group-hover:animate-bounce" />
                Watch Demo
              </Button>
            </div>
            
            <div className="flex items-center justify-center sm:justify-start space-x-8 pt-8">
              <div className="text-center group cursor-pointer">
                <div className="text-2xl font-bold text-primary group-hover:animate-bounce-subtle transition-all duration-300 group-hover:scale-110">99.9%</div>
                <div className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">Accuracy</div>
              </div>
              <div className="text-center group cursor-pointer">
                <div className="text-2xl font-bold text-primary group-hover:animate-bounce-subtle transition-all duration-300 group-hover:scale-110">24/7</div>
                <div className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">Monitoring</div>
              </div>
              <div className="text-center group cursor-pointer">
                <div className="text-2xl font-bold text-primary group-hover:animate-bounce-subtle transition-all duration-300 group-hover:scale-110">Real-time</div>
                <div className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">Detection</div>
              </div>
            </div>
          </div>
          
          <div className="relative animate-fade-in" style={{ animationDelay: '0.5s' }}>
            <div className="relative bg-card/50 backdrop-blur-sm rounded-2xl p-8 border border-steel-gray/30 hover:border-primary/30 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-metallic opacity-5 rounded-2xl group-hover:opacity-10 transition-opacity duration-500"></div>
              <div className="relative z-10">
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-primary/10 rounded-xl p-6 border border-primary/20 hover:border-primary/40 transition-all duration-300 group cursor-pointer hover:scale-105">
                    <Eye className="h-8 w-8 text-primary mb-4 group-hover:animate-bounce" />
                    <h3 className="font-semibold text-foreground mb-2">Machinery Tracking</h3>
                    <p className="text-sm text-muted-foreground">YOLOv8 powered detection</p>
                  </div>
                  <div className="bg-metallic/10 rounded-xl p-6 border border-metallic/20 hover:border-metallic/40 transition-all duration-300 group cursor-pointer hover:scale-105">
                    <Shield className="h-8 w-8 text-metallic mb-4 group-hover:animate-bounce" />
                    <h3 className="font-semibold text-foreground mb-2">Danger Zone Intrusion</h3>
                    <p className="text-sm text-muted-foreground">Spatial & centroid mapping</p>
                  </div>
                  <div className="bg-primary/10 rounded-xl p-6 border border-primary/20 hover:border-primary/40 transition-all duration-300 group cursor-pointer hover:scale-105">
                    <Zap className="h-8 w-8 text-primary mb-4 group-hover:animate-bounce" />
                    <h3 className="font-semibold text-foreground mb-2">Fall & Posture Detection</h3>
                    <p className="text-sm text-muted-foreground">Instant safety alerts</p>
                  </div>
                  <div className="bg-metallic/10 rounded-xl p-6 border border-metallic/20 hover:border-metallic/40 transition-all duration-300 group cursor-pointer hover:scale-105">
                    <Play className="h-8 w-8 text-metallic mb-4 group-hover:animate-bounce" />
                    <h3 className="font-semibold text-foreground mb-2">Command Dashboard</h3>
                    <p className="text-sm text-muted-foreground">Real-time site feeds</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;