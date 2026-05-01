import TireClassifier from "@/components/TireClassifier";

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-100 px-6 py-10">
      <section className="mx-auto mb-10 max-w-6xl text-center">
        <div className="mb-4 inline-flex rounded-full bg-orange-100 px-4 py-2 text-sm font-medium text-orange-700">
          FastAPI + Next.js + PyTorch
        </div>

        <h1 className="text-4xl font-bold tracking-tight text-zinc-950 md:text-6xl">
          Tire Safety Classifier
        </h1>

        <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-zinc-600">
          A full-stack machine learning application that detects tire condition
          from images using a ResNet50-based PyTorch model, served through
          FastAPI and displayed in a modern Next.js interface.
        </p>
      </section>

      <TireClassifier />
    </main>
  );
}