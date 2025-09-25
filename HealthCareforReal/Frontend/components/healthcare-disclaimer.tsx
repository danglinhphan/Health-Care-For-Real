'use client';

import { useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Checkbox } from '@/components/ui/checkbox';
import { AlertTriangle, Shield, UserCheck } from 'lucide-react';

interface HealthcareDisclaimerProps {
  isOpen: boolean;
  onAccept: () => void;
  onDecline: () => void;
}

export function HealthcareDisclaimer({ isOpen, onAccept, onDecline }: HealthcareDisclaimerProps) {
  const [acknowledged, setAcknowledged] = useState(false);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onDecline();
    }
  };

  return (
    <Dialog 
      open={isOpen} 
      onOpenChange={() => {}}
      aria-describedby="disclaimer-content"
    >
      <DialogContent 
        className="max-w-2xl max-h-[80vh] overflow-y-auto"
        onKeyDown={handleKeyDown}
      >
        <DialogHeader>
          <DialogTitle 
            className="flex items-center gap-2 text-red-600"
            id="disclaimer-title"
          >
            <AlertTriangle className="h-5 w-5" aria-hidden="true" />
            Important Healthcare Disclaimer
          </DialogTitle>
          <DialogDescription id="disclaimer-content">
            Please read and acknowledge the following important information before using this healthcare chatbot.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4" role="main" aria-labelledby="disclaimer-title">
          <Alert variant="destructive" role="alert" aria-live="assertive">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            <AlertDescription className="font-semibold">
              This chatbot is NOT a substitute for professional medical advice, diagnosis, or treatment.
            </AlertDescription>
          </Alert>

          <div className="space-y-3 text-sm">
            <div className="flex gap-3">
              <Shield className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold mb-1">Medical Advice Limitation</h4>
                <p className="text-gray-600">
                  This AI chatbot provides general health information only and cannot replace professional medical consultation. 
                  Always seek advice from qualified healthcare providers for medical concerns.
                </p>
              </div>
            </div>

            <div className="flex gap-3">
              <UserCheck className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold mb-1">Emergency Situations</h4>
                <p className="text-gray-600">
                  <strong>In case of medical emergency, call emergency services immediately (911 in US).</strong> 
                  Do not use this chatbot for urgent medical situations.
                </p>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">By using this chatbot, you acknowledge that:</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li>This tool provides educational information, not medical advice</li>
                <li>You will consult healthcare professionals for medical decisions</li>
                <li>The AI may make errors or provide incomplete information</li>
                <li>Your conversations may be stored for service improvement</li>
                <li>This service does not create a doctor-patient relationship</li>
              </ul>
            </div>

            <Alert>
              <AlertDescription>
                <strong>Privacy Notice:</strong> Your conversations are stored securely and used only to improve the service. 
                Do not share sensitive personal health information unnecessarily.
              </AlertDescription>
            </Alert>
          </div>

          <div className="flex items-center space-x-2 pt-4">
            <Checkbox 
              id="acknowledge" 
              checked={acknowledged}
              onCheckedChange={setAcknowledged}
              aria-describedby="acknowledge-description"
            />
            <label 
              htmlFor="acknowledge" 
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              id="acknowledge-description"
            >
              I have read, understood, and agree to the above disclaimer and terms
            </label>
          </div>
        </div>

        <DialogFooter>
          <Button 
            variant="outline" 
            onClick={onDecline}
            aria-label="Decline terms and exit"
          >
            Decline
          </Button>
          <Button 
            onClick={onAccept} 
            disabled={!acknowledged}
            className="bg-blue-600 hover:bg-blue-700"
            aria-label={acknowledged ? "Accept terms and continue to application" : "Please read and acknowledge the disclaimer first"}
          >
            Accept & Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function HealthcareDisclaimerBanner() {
  return (
    <Alert className="mb-4 border-amber-200 bg-amber-50">
      <AlertTriangle className="h-4 w-4 text-amber-600" />
      <AlertDescription className="text-amber-800">
        <strong>Medical Disclaimer:</strong> This chatbot provides general health information only. 
        Always consult qualified healthcare providers for medical advice. In emergencies, call 911.
      </AlertDescription>
    </Alert>
  );
}