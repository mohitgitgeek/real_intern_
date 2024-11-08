import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Shield, ShieldAlert, Info } from 'lucide-react';

const InternshipChecker = () => {
  const [announcement, setAnnouncement] = useState('');
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Simplified rules-based checking (since we can't include the actual ML model)
  const analyzeAnnouncement = () => {
    setIsAnalyzing(true);
    
    // Simulate processing time
    setTimeout(() => {
      const lowerText = announcement.toLowerCase();
      const redFlags = [
        'fee',
        'payment required',
        'pay to',
        'processing fee',
        'registration fee',
        'small fee',
        'urgent hiring',
        'guaranteed placement'
      ];
      
      const hasRedFlags = redFlags.some(flag => lowerText.includes(flag));
      const result = {
        isGenuine: !hasRedFlags,
        confidence: hasRedFlags ? 0.85 : 0.78,
        flags: redFlags.filter(flag => lowerText.includes(flag))
      };
      
      setResult(result);
      setIsAnalyzing(false);
    }, 1000);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-6 w-6" />
            Internship Announcement Checker
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Enter the internship announcement text:
            </label>
            <Textarea
              value={announcement}
              onChange={(e) => setAnnouncement(e.target.value)}
              placeholder="Paste the internship announcement here..."
              className="min-h-[120px]"
            />
          </div>
          
          <Button 
            onClick={analyzeAnnouncement}
            disabled={!announcement.trim() || isAnalyzing}
            className="w-full"
          >
            {isAnalyzing ? 'Analyzing...' : 'Check Announcement'}
          </Button>

          {result && (
            <div className="space-y-4">
              <Alert variant={result.isGenuine ? 'default' : 'destructive'}>
                <div className="flex items-center gap-2">
                  {result.isGenuine ? (
                    <Shield className="h-4 w-4" />
                  ) : (
                    <ShieldAlert className="h-4 w-4" />
                  )}
                  <AlertTitle>
                    {result.isGenuine ? 'Likely Genuine' : 'Potential Scam'}
                  </AlertTitle>
                </div>
                <AlertDescription>
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </AlertDescription>
              </Alert>

              {result.flags.length > 0 && (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Detected Red Flags:</AlertTitle>
                  <AlertDescription>
                    <ul className="list-disc pl-5 mt-2">
                      {result.flags.map((flag, index) => (
                        <li key={index} className="text-sm">
                          Contains "{flag}"
                        </li>
                      ))}
                    </ul>
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Tips for Identifying Genuine Internships</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            <li>• Legitimate internships never require payment or fees</li>
            <li>• Be cautious of urgency or pressure tactics</li>
            <li>• Verify the company's existence and contact information</li>
            <li>• Check if the email domain matches the official company website</li>
            <li>• Research the company on professional networking sites</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default InternshipChecker;