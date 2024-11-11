import React, { useState } from 'react';

interface DemoUploaderProps {
  onAnalysisComplete?: (analysis: any) => void;
}

export const DemoUploader: React.FC<DemoUploaderProps> = ({ onAnalysisComplete }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('demo', file);

    try {
      const response = await fetch('http://localhost:8000/analyze-demo', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze demo');
      }

      const data = await response.json();
      onAnalysisComplete?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-800 rounded-lg">
      <div className="flex flex-col gap-4">
        <div>
          <label 
            htmlFor="demo-file" 
            className="block text-sm font-medium text-gray-200 mb-2"
          >
            Upload Demo File (.dem)
          </label>
          <input
            id="demo-file"
            type="file"
            accept=".dem"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-500 file:text-white
                     hover:file:bg-blue-600
                     cursor-pointer"
          />
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleUpload}
            disabled={!file || isUploading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg
                     font-medium hover:bg-blue-600 disabled:opacity-50
                     disabled:cursor-not-allowed transition-colors"
          >
            {isUploading ? 'Analyzing...' : 'Analyze Demo'}
          </button>

          {file && (
            <span className="text-sm text-gray-300">
              Selected: {file.name}
            </span>
          )}
        </div>

        {error && (
          <div className="text-red-400 text-sm">
            Error: {error}
          </div>
        )}
      </div>
    </div>
  );
};