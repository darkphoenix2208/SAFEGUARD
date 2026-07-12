import { Eye, Users, Activity, Package, AlertTriangle, Search, Camera, Brain, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";

const Features = () => {
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  const features = [
    {
      title: "Danger Zone Intrusion",
      description: "Advanced AI algorithms identify suspicious danger_zone_intrusion behavior in restricted areas with precision timing",
      icon: Search,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
      borderColor: "border-blue-500/20",
      delay: "0s"
    },
    {
      title: "Motion Detection",
      description: "Intelligent motion analysis during after-hours periods with customizable sensitivity settings",
      icon: Activity,
      color: "text-green-500",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/20",
      delay: "0.1s"
    },
    {
      title: "Shadow Analysis",
      description: "Sophisticated shadow detection and analysis during midnight hours for enhanced security",
      icon: Eye,
      color: "text-purple-500",
      bgColor: "bg-purple-500/10",
      borderColor: "border-purple-500/20",
      delay: "0.2s"
    },
    {
      title: "YOLOv8 Object Detection",
      description: "State-of-the-art real-time object recognition and classification with 99.9% accuracy",
      icon: Camera,
      color: "text-orange-500",
      bgColor: "bg-orange-500/10",
      borderColor: "border-orange-500/20",
      delay: "0.3s"
    },
    {
      title: "Fall from Height Detection",
      description: "Immediate emergency alerts for fall incidents from height requiring urgent medical attention",
      icon: AlertTriangle,
      color: "text-red-500",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/20",
      delay: "0.4s"
    },
    {
      title: "Person Tracking",
      description: "Centroid-based tracking system assigns unique IDs for comprehensive entry-exit monitoring",
      icon: Users,
      color: "text-cyan-500",
      bgColor: "bg-cyan-500/10",
      borderColor: "border-cyan-500/20",
      delay: "0.5s"
    },
    {
      title: "Pose Estimation",
      description: "MediaPipe-powered pose analysis detecting sitting/standing positions for behavioral monitoring",
      icon: Brain,
      color: "text-pink-500",
      bgColor: "bg-pink-500/10",
      borderColor: "border-pink-500/20",
      delay: "0.6s"
    },
    {
      title: "Abandoned Object Detection",
      description: "Smart detection of objects left unattended for extended periods with customizable thresholds",
      icon: Package,
      color: "text-yellow-500",
      bgColor: "bg-yellow-500/10",
      borderColor: "border-yellow-500/20",
      delay: "0.7s"
    }
  ];

  return (
    <section id="features" className="py-24 bg-gradient-hero relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-64 h-64 bg-primary/5 rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-metallic/5 rounded-full blur-3xl animate-float" style={{ animationDelay: '3s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-primary/3 rounded-full blur-3xl animate-float" style={{ animationDelay: '1.5s' }}></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 text-primary mb-4">
            <Zap className="h-6 w-6 animate-pulse" />
            <span className="text-sm font-medium uppercase tracking-wider">AI-Powered Features</span>
          </div>
          <h2 className="text-4xl lg:text-6xl font-bold mb-6">
            <span className="bg-gradient-metallic bg-clip-text text-transparent">
              Advanced Surveillance
            </span>
            <br />
            <span className="text-foreground">Capabilities</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Experience the future of security with our comprehensive AI-driven surveillance features
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const FeatureIcon = feature.icon;
            const isHovered = hoveredCard === index;
            
            return (
              <Card 
                key={index}
                className={`
                  group relative bg-card/80 backdrop-blur-sm border transition-all duration-500 ease-out cursor-pointer
                  ${isHovered ? 'shadow-2xl shadow-primary/20 scale-105' : 'shadow-lg'}
                  ${feature.borderColor}
                  hover:border-primary/30
                `}
                style={{ 
                  animationDelay: feature.delay,
                  animation: `slide-up 0.6s ease-out ${feature.delay} both`
                }}
                onMouseEnter={() => setHoveredCard(index)}
                onMouseLeave={() => setHoveredCard(null)}
              >
                {/* Shine effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out"></div>
                
                <CardHeader className="pb-4">
                  <div className={`
                    p-3 rounded-xl w-fit transition-all duration-300
                    ${feature.bgColor}
                    ${isHovered ? 'scale-110 animate-bounce-subtle' : ''}
                  `}>
                    <FeatureIcon className={`h-6 w-6 ${feature.color} transition-all duration-300`} />
                  </div>
                  <CardTitle className={`
                    text-lg text-foreground transition-all duration-300
                    ${isHovered ? 'text-primary' : ''}
                  `}>
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className={`
                    text-muted-foreground leading-relaxed transition-all duration-300
                    ${isHovered ? 'text-foreground/80' : ''}
                  `}>
                    {feature.description}
                  </CardDescription>
                </CardContent>

                {/* Glow effect on hover */}
                {isHovered && (
                  <div className="absolute inset-0 bg-primary/5 rounded-lg blur-xl animate-pulse-glow"></div>
                )}
              </Card>
            );
          })}
        </div>

        {/* Technical Specifications */}
        <div className="mt-20 animate-fade-in" style={{ animationDelay: '1s' }}>
          <div className="bg-card/50 backdrop-blur-sm rounded-2xl p-8 border border-steel-gray/30 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-metallic opacity-5 rounded-2xl"></div>
            <div className="relative z-10">
              <h3 className="text-3xl font-bold text-center mb-8 text-foreground">
                Technical Excellence
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                <div className="text-center group hover:scale-105 transition-transform duration-300">
                  <div className="text-4xl font-bold text-primary mb-2 group-hover:animate-bounce">YOLOv8</div>
                  <p className="text-muted-foreground">Latest AI Model</p>
                </div>
                <div className="text-center group hover:scale-105 transition-transform duration-300">
                  <div className="text-4xl font-bold text-primary mb-2 group-hover:animate-bounce">MediaPipe</div>
                  <p className="text-muted-foreground">Pose Detection</p>
                </div>
                <div className="text-center group hover:scale-105 transition-transform duration-300">
                  <div className="text-4xl font-bold text-primary mb-2 group-hover:animate-bounce">99.9%</div>
                  <p className="text-muted-foreground">Accuracy Rate</p>
                </div>
                <div className="text-center group hover:scale-105 transition-transform duration-300">
                  <div className="text-4xl font-bold text-primary mb-2 group-hover:animate-bounce">&lt;1s</div>
                  <p className="text-muted-foreground">Response Time</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;