export type ScentResponse = {
  status: string;
  screenshot_size: {
    width: number;
    height: number;
  };
  cartridges: Record<string, number>;
  percentages: Record<string, number>;
  pwm: Record<string, number>;
  sources: Record<string, number>;
  scene_areas: Record<string, number>;
  should_send: boolean;
};

export async function analyzeScreen(): Promise<ScentResponse> {
  const response = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Backend error: ${response.status}`);
  }

  return response.json();
}