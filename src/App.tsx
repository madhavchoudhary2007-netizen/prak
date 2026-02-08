import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Leaf, Search, X, Send, BookOpen, Users, Zap, Bot, Brain } from 'lucide-react';

interface Plant {
  id: number;
  name: string;
  sciName: string;
  image: string;
  description: string;
  ethnobotanical: string;
  medicinalUses: string[];
  bioelectric: string;
  ailments: string[];
}

const plants: Plant[] = [
  {
    id: 1,
    name: "Tulsi",
    sciName: "Ocimum tenuiflorum",
    image: "/images/tulsi.jpg",
    description: "Known as the 'Queen of Herbs' in Ayurveda, Tulsi is a sacred plant revered for its spiritual and medicinal properties. It thrives in the tropical climates of India.",
    ethnobotanical: "Deeply embedded in Hindu traditions and tribal practices across India. Used in rituals for purification and protection by communities in Uttar Pradesh, Bihar, and the Western Ghats.",
    medicinalUses: ["Respiratory infections", "Stress and anxiety relief", "Immune system boost", "Anti-inflammatory effects"],
    bioelectric: "Essential oils like eugenol help stabilize neural bioelectric signals, acting as a natural nervine tonic that calms overactive nerve impulses similar to a biofeedback mechanism.",
    ailments: ["respiratory", "stress", "immunity"]
  },
  {
    id: 2,
    name: "Neem",
    sciName: "Azadirachta indica",
    image: "/images/neem.jpg",
    description: "A versatile evergreen tree native to the Indian subcontinent, famous for its bitter leaves and powerful antimicrobial properties.",
    ethnobotanical: "Used by indigenous tribes in Rajasthan and Madhya Pradesh for oral hygiene, skin care, and as natural pesticides in farming rituals.",
    medicinalUses: ["Skin disorders", "Dental health", "Blood purification", "Antibacterial action"],
    bioelectric: "Azadirachtin compounds influence cellular membrane potentials, modulating ion channels akin to electrical gating in bioelectronic systems.",
    ailments: ["skin"]
  },
  {
    id: 3,
    name: "Ashwagandha",
    sciName: "Withania somnifera",
    image: "/images/ashwagandha.jpg",
    description: "A powerful adaptogen known as 'Indian Ginseng', used for over 3,000 years to combat stress and promote vitality.",
    ethnobotanical: "Integral to traditional healing in the deserts of Rajasthan and Himalayan foothills, passed down in Ayurvedic texts and folk practices.",
    medicinalUses: ["Stress reduction", "Cognitive enhancement", "Energy and vitality", "Hormonal balance"],
    bioelectric: "Withanolides regulate GABAergic neurotransmission, mimicking bioelectric dampening to reduce anxiety-related neural firing patterns.",
    ailments: ["stress", "memory"]
  },
  {
    id: 4,
    name: "Turmeric",
    sciName: "Curcuma longa",
    image: "/images/turmeric.jpg",
    description: "The golden spice of India, containing curcumin which gives it its vibrant color and potent healing abilities.",
    ethnobotanical: "Central to wedding rituals, festivals, and healing ceremonies in Tamil Nadu, Kerala, and Bengal communities.",
    medicinalUses: ["Anti-inflammatory", "Wound healing", "Digestive aid", "Antioxidant protection"],
    bioelectric: "Curcumin modulates voltage-gated ion channels in nerves, providing neuroprotective effects comparable to signal filtering in electronic circuits.",
    ailments: ["inflammation", "digestion", "skin"]
  },
  {
    id: 5,
    name: "Ginger",
    sciName: "Zingiber officinale",
    image: "/images/ginger.jpg",
    description: "A pungent rhizome used worldwide but deeply rooted in Indian cuisine and medicine for its warming properties.",
    ethnobotanical: "Employed by fishing communities in Kerala and coastal tribes for nausea and as a digestive tonic in monsoon rituals.",
    medicinalUses: ["Nausea and motion sickness", "Digestive disorders", "Cold and flu relief", "Anti-inflammatory"],
    bioelectric: "Gingerols enhance gastric motility through bioelectric stimulation of smooth muscle cells, similar to pacemaker impulses.",
    ailments: ["digestion", "respiratory"]
  },
  {
    id: 6,
    name: "Amla",
    sciName: "Phyllanthus emblica",
    image: "/images/amla.jpg",
    description: "The Indian gooseberry, a superfruit packed with Vitamin C and revered as a Rasayana (rejuvenator) in Ayurveda.",
    ethnobotanical: "Harvested during festivals by Adivasi communities in Central India, symbolizing longevity in folk tales.",
    medicinalUses: ["Immune support", "Hair and skin health", "Digestive health", "Blood sugar regulation"],
    bioelectric: "High antioxidant content protects neuronal membranes, preserving bioelectric integrity against oxidative stress.",
    ailments: ["immunity", "digestion", "skin"]
  },
  {
    id: 7,
    name: "Brahmi",
    sciName: "Bacopa monnieri",
    image: "/images/brahmi.jpg",
    description: "A creeping herb thriving in wetlands, traditionally used to sharpen memory and intellect.",
    ethnobotanical: "Sacred to students in ancient gurukuls, used by communities in Odisha and Kerala for cognitive rituals.",
    medicinalUses: ["Memory enhancement", "Anxiety reduction", "Cognitive function", "Neuroprotection"],
    bioelectric: "Bacosides promote synaptic plasticity by enhancing dendritic spine density and modulating sodium-potassium pumps.",
    ailments: ["memory", "stress"]
  },
  {
    id: 8,
    name: "Shankhpushpi",
    sciName: "Convolvulus pluricaulis",
    image: "/images/shankhpushpi.jpg",
    description: "Known as the 'shankh' flower, this delicate herb is a premier brain tonic in traditional Indian medicine.",
    ethnobotanical: "Utilized by Rajasthani and Gujarati folk healers for mental clarity and in spiritual meditation practices.",
    medicinalUses: ["Mental clarity", "Insomnia relief", "Nervous disorders", "Cognitive support"],
    bioelectric: "Enhances acetylcholine activity, facilitating smoother neural impulse transmission akin to circuit optimization.",
    ailments: ["memory", "stress"]
  },
  {
    id: 9,
    name: "Jatamansi",
    sciName: "Nardostachys jatamansi",
    image: "/images/jatamansi.jpg",
    description: "A rare Himalayan herb with a musky aroma, prized for its calming and grounding effects on the mind.",
    ethnobotanical: "Gathered by high-altitude tribes in Uttarakhand for shamanic ceremonies and treating mental fatigue.",
    medicinalUses: ["Calming the mind", "Sleep disorders", "Nervine tonic", "Emotional balance"],
    bioelectric: "Valerenic acid analogs inhibit excessive neural firing, providing a bioelectric sedative effect on the central nervous system.",
    ailments: ["stress", "memory"]
  },
  {
    id: 10,
    name: "Shatavari",
    sciName: "Asparagus racemosus",
    image: "/images/shatavari.jpg",
    description: "The 'queen of herbs' for women's health, this climbing plant supports reproductive wellness and vitality.",
    ethnobotanical: "Cherished in folk traditions of Maharashtra and Karnataka for maternal health and as an aphrodisiac.",
    medicinalUses: ["Hormonal balance", "Reproductive health", "Lactation support", "Immune modulation"],
    bioelectric: "Saponins influence estrogen receptor signaling pathways, subtly affecting bioelectric patterns in endocrine tissues.",
    ailments: ["digestion", "immunity"]
  },
  {
    id: 11,
    name: "Guduchi",
    sciName: "Tinospora cordifolia",
    image: "/images/guduchi.jpg",
    description: "Called 'Amrita' or nectar of immortality, this climbing vine is a cornerstone of Ayurvedic immunity boosters.",
    ethnobotanical: "Used by tribal groups in the Eastern Ghats for treating fevers and as a tonic during seasonal changes.",
    medicinalUses: ["Fever reduction", "Immune enhancement", "Detoxification", "Joint health"],
    bioelectric: "Polysaccharides modulate T-cell activity, supporting bioelectric signaling in immune response cascades.",
    ailments: ["immunity", "inflammation"]
  },
  {
    id: 12,
    name: "Haritaki",
    sciName: "Terminalia chebula",
    image: "/images/haritaki.jpg",
    description: "One of the three fruits in Triphala, revered as the 'king of medicines' for its detoxifying and rejuvenating powers.",
    ethnobotanical: "Pivotal in Tibetan and Ayurvedic traditions, used by Himalayan communities in longevity elixirs.",
    medicinalUses: ["Detoxification", "Digestive health", "Antimicrobial", "Eye health"],
    bioelectric: "Tannins help regulate gut-brain axis signals, influencing vagus nerve bioelectric impulses for better digestion.",
    ailments: ["digestion", "respiratory"]
  }
];

const ailmentLabels: Record<string, string> = {
  "all": "All Plants",
  "stress": "Stress & Anxiety",
  "memory": "Cognitive Health",
  "respiratory": "Respiratory Relief",
  "digestion": "Digestive Wellness",
  "immunity": "Immunity Boost",
  "skin": "Skin & Beauty",
  "inflammation": "Anti-Inflammatory"
};

function App() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeFilter, setActiveFilter] = useState("all");
  const [selectedPlant, setSelectedPlant] = useState<Plant | null>(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'bot'; content: string; meta?: { intent: string; confidence: number; entities: Record<string, string[]> } }[]>([
    { role: 'bot', content: "Namaste! I am Vaidya, your AI-powered ethnobotany guide. I use Natural Language Processing to understand your questions about Indian medicinal plants, their traditional uses, bioelectric properties, and more.\n\nTry asking:\n- 'Tell me about Ashwagandha'\n- 'Which plants help with stress?'\n- 'What are nervine tonics?'\n- 'Bioelectric properties of Brahmi'" }
  ]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const filteredPlants = plants.filter(plant => {
    const matchesSearch = plant.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         plant.sciName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         plant.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesFilter = activeFilter === "all" || 
                         plant.ailments.includes(activeFilter);
    
    return matchesSearch && matchesFilter;
  });

  const API_BASE = 'http://localhost:5000';

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInputMessage("");
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();
      setMessages(prev => [...prev, {
        role: 'bot',
        content: data.response,
        meta: {
          intent: data.intent,
          confidence: data.confidence,
          entities: data.entities,
        }
      }]);
    } catch {
      // Fallback to local rule-based response if backend is unavailable
      const fallbackReply = getLocalFallbackResponse(userMessage);
      setMessages(prev => [...prev, {
        role: 'bot',
        content: fallbackReply,
        meta: { intent: 'fallback', confidence: 0, entities: {} }
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const getLocalFallbackResponse = (query: string): string => {
    const lq = query.toLowerCase().trim();
    const matchedPlant = plants.find(p =>
      lq.includes(p.name.toLowerCase()) ||
      lq.includes(p.sciName.toLowerCase().split(' ')[0])
    );
    if (matchedPlant) {
      return `**${matchedPlant.name} (${matchedPlant.sciName})**\n\n${matchedPlant.description}\n\n**Traditional Uses:** ${matchedPlant.medicinalUses.join(', ')}\n\n**Bioelectric Properties:** ${matchedPlant.bioelectric}`;
    }
    if (lq.includes("hello") || lq.includes("namaste") || lq.includes("hi")) {
      return "Namaste! I'm currently running in offline mode. Ask me about any Indian medicinal plant!";
    }
    return "I'm currently running in offline mode (backend unavailable). Please ensure the Python AI backend is running on port 5000. Try: `cd backend && python app.py`";
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const openPlantModal = (plant: Plant) => {
    setSelectedPlant(plant);
  };

  const closePlantModal = () => {
    setSelectedPlant(null);
  };

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
    if (!isChatOpen && messages.length === 1) {
      // Welcome already there
    }
  };

  return (
    <div className="min-h-screen bg-[#0A120F] text-white overflow-hidden">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0A120F]/95 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-[#E8B059] flex items-center justify-center">
                <Leaf className="w-5 h-5 text-[#0A120F]" />
              </div>
              <div>
                <div className="font-serif text-2xl tracking-[3px]">PRAKRITI</div>
                <div className="text-[10px] text-white/60 -mt-1">ETHNOBOTANICAL AI</div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-10 text-sm uppercase tracking-[2px]">
            <a href="#discover" className="hover:text-[#E8B059] transition-colors">Discover</a>
            <a href="#about" className="hover:text-[#E8B059] transition-colors">The Science</a>
            <button 
              onClick={toggleChat}
              className="flex items-center gap-2 px-6 py-2.5 rounded-full border border-white/30 hover:bg-white/5 transition-all active:scale-[0.985]"
            >
              <span>CHAT WITH VAIDYA</span>
              <Leaf className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center pt-20 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(#ffffff10_0.8px,transparent_1px)] bg-[length:4px_4px]"></div>
        
        <div className="absolute inset-0 bg-[linear-gradient(to_bottom,#0A120F_40%,transparent)] z-10"></div>

        <div className="relative z-20 text-center px-6 max-w-5xl">
          <motion.div 
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
            className="mb-8 inline-flex items-center gap-3 px-4 py-1 rounded-full border border-white/20 text-xs tracking-[3px]"
          >
            PRESERVING INDIA'S LIVING HERITAGE
          </motion.div>
          
          <h1 className="font-serif text-[92px] leading-[0.92] tracking-[-4.2px] mb-8">
            ANCIENT<br />WISDOM<br />MEETS<br />BIOELECTRONICS
          </h1>
          
          <p className="max-w-[560px] mx-auto text-xl text-white/70 mb-14">
            Explore 3,000 years of Indian ethnobotanical knowledge. Discover medicinal plants, 
            their cultural significance, and the hidden bioelectric mechanisms behind their healing power.
          </p>

          <div className="flex justify-center gap-4">
            <a href="#discover" 
               className="group flex items-center justify-center gap-3 px-10 py-4 rounded-full bg-white text-black hover:bg-[#E8B059] font-medium transition-all active:scale-[0.985]">
              EXPLORE THE GARDEN
              <motion.div animate={{ rotate: 15 }} className="group-hover:rotate-45 transition">
                <Leaf className="w-4 h-4" />
              </motion.div>
            </a>
            
            <button 
              onClick={toggleChat}
              className="group flex items-center justify-center gap-3 px-10 py-4 rounded-full border border-white/40 hover:bg-white/5 transition-all active:scale-[0.985]">
              SPEAK WITH VAIDYA
            </button>
          </div>
        </div>

        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 z-20">
          <div className="text-[10px] tracking-[3px] text-white/50">SCROLL TO BEGIN</div>
          <motion.div animate={{ y: [0, 12, 0] }} transition={{ duration: 2.2, repeat: Infinity }}>
            <div className="h-px w-px bg-white/40"></div>
          </motion.div>
        </div>
      </section>

      {/* Discover Section */}
      <section id="discover" className="max-w-7xl mx-auto px-6 pb-24 pt-12">
        <div className="flex flex-col md:flex-row justify-between items-end mb-14">
          <div>
            <div className="uppercase tracking-[4px] text-xs text-[#E8B059]">DATABASE</div>
            <h2 className="text-7xl font-serif tracking-tighter mt-1">Living Library</h2>
          </div>
          
          <div className="mt-8 md:mt-0 flex items-center gap-4">
            <div className="relative w-80">
              <input
                type="text"
                placeholder="Search plants..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-transparent border border-white/20 rounded-full pl-12 py-4 text-lg placeholder:text-white/50 focus:outline-none focus:border-[#E8B059]"
              />
              <Search className="absolute left-5 top-4.5 text-white/50" />
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 pb-10 border-b border-white/10">
          {Object.keys(ailmentLabels).map((key) => (
            <button
              key={key}
              onClick={() => setActiveFilter(key)}
              className={`px-8 py-3 text-sm rounded-full border transition-all ${activeFilter === key 
                ? "bg-[#E8B059] text-black border-[#E8B059]" 
                : "border-white/20 hover:border-white/60"}`}
            >
              {ailmentLabels[key]}
            </button>
          ))}
        </div>

        {/* Plants Grid */}
        <div className="pt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <AnimatePresence>
            {filteredPlants.length > 0 ? (
              filteredPlants.map((plant, index) => (
                <motion.div
                  key={plant.id}
                  initial={{ opacity: 0, y: 80 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 40 }}
                  transition={{ delay: Math.min(index * 0.035, 0.6) }}
                  onClick={() => openPlantModal(plant)}
                  className="group cursor-pointer overflow-hidden rounded-3xl bg-[#111815] border border-white/10 hover:border-[#E8B059]/50 transition-all duration-500"
                >
                  <div className="relative h-[390px] overflow-hidden">
                    <img 
                      src={plant.image} 
                      alt={plant.name} 
                      className="absolute inset-0 w-full h-full object-cover scale-[1.01] group-hover:scale-110 transition-transform duration-[1200ms]"
                    />
                    <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-black/60 to-black/90" />
                    
                    <div className="absolute bottom-0 left-0 right-0 p-8">
                      <div className="flex justify-between items-end">
                        <div>
                          <div className="text-[#E8B059] text-xs tracking-[2px] mb-1">INDIAN FLORA</div>
                          <div className="font-serif text-5xl tracking-[-1.5px]">{plant.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs opacity-60">SCIENTIFIC NAME</div>
                          <div className="font-mono text-xs text-[#E8B059]">{plant.sciName}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-8 pt-7 flex flex-col">
                    <p className="line-clamp-3 text-white/70 text-[15px] leading-snug">{plant.description}</p>
                    
                    <div className="mt-auto pt-8 flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-[#E8B059]">
                        <BookOpen className="w-4 h-4" /> TRADITIONAL
                      </div>
                      <div className="text-white/60 group-hover:text-white transition">VIEW PROFILE →</div>
                    </div>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="col-span-full py-20 text-center">
                <div className="mx-auto w-16 h-16 rounded-full border border-white/10 flex items-center justify-center mb-6">
                  <Search className="w-8 h-8 text-white/40" />
                </div>
                <p className="text-xl text-white/60">No plants found matching your search.</p>
                <button onClick={() => {setSearchTerm(""); setActiveFilter("all");}} className="mt-4 underline text-sm">CLEAR FILTERS</button>
              </div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* Interdisciplinary Section */}
      <section id="about" className="bg-black py-24 border-t border-white/10">
        <div className="max-w-5xl mx-auto px-6">
          <div className="grid md:grid-cols-12 gap-x-6">
            <div className="md:col-span-5">
              <div className="sticky top-24">
                <div className="uppercase text-xs tracking-[3px] text-[#E8B059]">INTERDISCIPLINARY BRIDGE</div>
                <h3 className="text-[68px] leading-none tracking-tighter mt-6 font-serif">Where Ancient<br />Healing Meets<br />Modern Science</h3>
              </div>
            </div>

            <div className="md:col-span-7 mt-16 md:mt-0 text-[17px] text-white/75 leading-relaxed space-y-9">
              <p>The project bridges the knowledge gap between traditional ethnobotanical descriptions and modern bioengineering. Many revered plants in Indian medicine have powerful effects on the nervous system that can be understood through bioelectric principles — the very same electrical signaling that powers our neurons.</p>
              
              <div className="pl-8 border-l-2 border-white/20 text-sm space-y-8 pt-1">
                <div className="flex gap-6">
                  <div><Zap className="w-5 h-5 mt-1 text-[#E8B059]" /></div>
                  <div>Nervine plants like Brahmi and Jatamansi modulate ion channel activity, helping regulate action potentials in neurons — a direct connection between traditional calming effects and bioelectronics.</div>
                </div>
                <div className="flex gap-6">
                  <div><Users className="w-5 h-5 mt-1 text-[#E8B059]" /></div>
                  <div>These plants preserve invaluable cultural knowledge from India’s diverse communities while offering exciting potential for the future of neurotechnology and natural therapeutics.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black py-20">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-y-16">
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-full bg-[#E8B059] flex items-center justify-center">
                <Leaf className="w-4 h-4 text-black" />
              </div>
              <span className="font-serif tracking-[1px] text-xl">PRAKRITI AI</span>
            </div>
            
            <p className="max-w-xs text-white/60">A student-built demonstration project celebrating Indian ethnobotany and the emerging field of bioelectric medicine.</p>
            
            <div className="mt-8 flex gap-8 text-xs text-white/50">
              <div>REACT + TAILWIND + PYTHON NLP</div>
              <div>AI-POWERED • TF-IDF + SVM CLASSIFIER</div>
            </div>
          </div>

          <div className="text-sm text-white/60 md:text-right space-y-2">
            <div>Disclaimer: This application is for educational and demonstration purposes only.</div>
            <div>Information presented is not a substitute for professional medical advice.</div>
            <div className="pt-4">© {new Date().getFullYear()} Prakriti AI. All Rights Preserved.</div>
          </div>
        </div>
      </footer>

      {/* Plant Detail Modal */}
      <AnimatePresence>
        {selectedPlant && (
          <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/90 p-6" onClick={closePlantModal}>
            <motion.div 
              initial={{ opacity: 0, scale: 0.96, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 30 }}
              transition={{ type: "spring", bounce: 0.02, duration: 0.4 }}
              onClick={e => e.stopPropagation()}
              className="bg-[#111815] max-w-[1000px] w-full rounded-3xl overflow-hidden relative"
            >
              <button onClick={closePlantModal} className="absolute top-8 right-8 z-10 text-white/70 hover:text-white">
                <X size={26} />
              </button>

              <div className="grid md:grid-cols-5 h-[640px]">
                {/* Left: Image */}
                <div className="md:col-span-3 relative overflow-hidden">
                  <img 
                    src={selectedPlant.image} 
                    alt={selectedPlant.name} 
                    className="absolute inset-0 w-full h-full object-cover" 
                  />
                  <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-transparent" />
                  
                  <div className="absolute bottom-0 p-12 text-white">
                    <div className="text-[#E8B059] text-sm tracking-[2px]">ETHNOBOTANICAL PROFILE</div>
                    <div className="mt-4 text-[76px] font-serif leading-none tracking-[-2px]">{selectedPlant.name}</div>
                    <div className="font-mono mt-2 text-xl opacity-75">{selectedPlant.sciName}</div>
                  </div>
                </div>

                {/* Right: Details */}
                <div className="md:col-span-2 p-12 md:pr-16 flex flex-col overflow-y-auto">
                  <div className="mb-auto">
                    <div className="uppercase tracking-widest text-xs mb-4 text-white/50">OVERVIEW</div>
                    <p className="text-xl leading-snug">{selectedPlant.description}</p>
                  </div>
                  
                  <div className="-mx-1 mt-12 space-y-12 text-sm">
                    <div>
                      <div className="font-medium tracking-[1px] text-[#E8B059] mb-4 flex items-center gap-3">ETHNOBOTANICAL CONTEXT <div className="h-px flex-1 bg-white/10" /></div>
                      <p className="text-white/80 leading-relaxed">{selectedPlant.ethnobotanical}</p>
                    </div>
                    
                    <div>
                      <div className="font-medium tracking-[1px] text-[#E8B059] mb-4 flex items-center gap-3">MEDICINAL APPLICATIONS <div className="h-px flex-1 bg-white/10" /></div>
                      <ul className="space-y-3 text-[15px]">
                        {selectedPlant.medicinalUses.map((use, idx) => (
                          <li key={idx} className="flex items-start gap-3">
                            <div className="-mt-px text-[#E8B059]">◉</div> {use}
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <div className="font-medium tracking-[1px] text-[#E8B059] mb-4 flex items-center gap-3">BIOELECTRIC INSIGHTS <div className="h-px flex-1 bg-white/10" /></div>
                      <p className="leading-relaxed text-[15px] text-white/80">{selectedPlant.bioelectric}</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Chat Modal */}
      <AnimatePresence>
        {isChatOpen && (
          <div className="fixed inset-0 z-[80] flex items-end justify-end p-4 md:p-8 pointer-events-none">
            <motion.div 
              initial={{ opacity: 0, scale: 0.96, y: 50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 60 }}
              className="bg-[#0F1513] w-full max-w-[440px] rounded-3xl shadow-2xl border border-white/10 overflow-hidden pointer-events-auto flex flex-col h-[640px]"
            >
              {/* Chat Header */}
              <div className="px-8 py-6 flex items-center justify-between border-b border-white/10">
                <div className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[#E8B059]/50">
                    <Bot className="text-[#E8B059]" />
                  </div>
                  <div>
                    <div className="font-medium">Vaidya • NLP-Powered</div>
                    <div className="text-xs text-emerald-400 flex items-center gap-1.5">● AI ENGINE ACTIVE</div>
                  </div>
                </div>
                <button onClick={toggleChat} className="text-white/70 hover:text-white">
                  <X size={22} />
                </button>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-8 space-y-8 text-[15px] custom-scroll">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[86%] ${msg.role === 'bot' ? '' : ''}`}>
                      <div
                        className={`px-6 py-4 rounded-3xl ${msg.role === 'user'
                          ? 'bg-[#E8B059] text-black'
                          : 'bg-white/5'}`}
                      >
                        {msg.content.split('\n').map((line, i) => (
                          <div key={i} className={line.startsWith('**') ? 'font-semibold mt-2' : ''}>
                            {line.replace(/\*\*/g, '').replace(/\*/g, '')}
                          </div>
                        ))}
                      </div>
                      {msg.role === 'bot' && msg.meta && msg.meta.intent !== 'fallback' && msg.meta.confidence > 0 && (
                        <div className="flex items-center gap-2 mt-1.5 px-2">
                          <Brain className="w-3 h-3 text-[#E8B059]/60" />
                          <span className="text-[10px] text-white/30">
                            NLP: {msg.meta.intent.replace(/_/g, ' ')} ({(msg.meta.confidence * 100).toFixed(0)}% confidence)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-white/5 px-6 py-4 rounded-3xl flex items-center gap-2">
                      <div className="flex gap-1">
                        <motion.div className="w-1 h-1 bg-white/60 rounded-full" animate={{ scale: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: 1.1, delay: 0 }} />
                        <motion.div className="w-1 h-1 bg-white/60 rounded-full" animate={{ scale: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: 1.1, delay: 0.2 }} />
                        <motion.div className="w-1 h-1 bg-white/60 rounded-full" animate={{ scale: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: 1.1, delay: 0.4 }} />
                      </div>
                      <span className="text-xs text-white/50 ml-2">AI Processing...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-white/10">
                <div className="flex rounded-2xl bg-white/[0.035] focus-within:bg-white/[0.07] border border-white/10 focus-within:border-white/30">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about a plant or ailment..."
                    className="flex-1 bg-transparent px-7 py-4 text-[15px] placeholder:text-white/50 focus:outline-none"
                  />
                  <button 
                    onClick={sendMessage} 
                    disabled={!inputMessage.trim()}
                    className="mr-2 my-1.5 px-4 flex items-center justify-center rounded-xl bg-[#E8B059] disabled:bg-white/10 disabled:text-white/40 text-black"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
                
                <div className="text-center mt-4 text-[10px] text-white/40">For educational purposes only. Not medical advice.</div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Floating Chat Button */}
      {!isChatOpen && (
        <button 
          onClick={toggleChat}
          className="fixed bottom-10 right-10 z-[60] flex h-16 w-16 items-center justify-center rounded-full bg-[#E8B059] text-black shadow-xl active:scale-95 transition-all hover:shadow-[#E8B059]/50"
        >
          <Leaf className="w-8 h-8" />
        </button>
      )}
    </div>
  );
}

export default App;
