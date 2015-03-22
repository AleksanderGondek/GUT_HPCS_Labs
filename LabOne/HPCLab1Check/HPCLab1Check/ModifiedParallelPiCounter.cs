using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

namespace HPCLab1Check
{
    sealed internal class ModifiedParallelPiCounter
    {
        public int NumberOfWorkers { get; set; }
        public double Precision { get; set; }
        private readonly ConcurrentStack<double> _fractionsStack;

        public ModifiedParallelPiCounter()
        {
            _fractionsStack = new ConcurrentStack<double>();
        }

        public double CountPi()
        {
            if (Precision < NumberOfWorkers)
            {
                throw new Exception("Precision smaller than the number of processes - try again.");
            }

            Parallel.For(0, NumberOfWorkers, i =>
            {
                // ---------------------- 
                // Sum( ( (-1)^n ) / (2*n+1) ) n=0 to Precision-1
                // ----------------------
                int myN = i;

                double myLocalSum = 0.0d;
                while (myN < Precision)
                {
                    var nThElement = ((Math.Pow(-1.0d, myN)) / ((2 * myN) + 1));
                    Console.WriteLine("Worker {0}; myN {1}; myStopN {2}; nThElement {3}", i, myN, Precision, nThElement);
                    myLocalSum += nThElement;
                    myN += NumberOfWorkers;
                }

                _fractionsStack.Push(myLocalSum);
            });

            var allFractions = _fractionsStack.ToArray().Sum();
            return allFractions*4;
        }
    }
}
