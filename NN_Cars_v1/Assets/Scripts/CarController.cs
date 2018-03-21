using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;


public class CarController : MonoBehaviour {

    [SerializeField]
    public bool run = true;
    public float maxTime = 20;
    public GameObject redCarPrefab;
    public GameObject greenCarPrefab;
    public GameObject blueCarPrefab;
    public GameObject orangeCarPrefab;
    public GameObject yellowCarPrefab;
    public GameObject ScorepointParent;
    public GameObject redCarParent;
    public GameObject greenCarParent;
    public GameObject blueCarParent;
    public GameObject orangeCarParent;
    public GameObject yellowCarParent;
    public int redCarCount;
    public int greenCarCount;
    public int blueCarCount;
    public int orangeCarCount;
    public Text score;
    public Text gen;

    private float time;
    private int generation;

    public static string[] colliders;

    
    List<CarControl> cars;
    List<CarControl> redCars;
    List<CarControl> greenCars;
    List<CarControl> blueCars;
    List<CarControl> orangeCars;
    CarControl yellowCar;

    // Use this for initialization
    void Start () {
        int count = ScorepointParent.transform.childCount;
        colliders = new string[count];
        for (int i = 0; i < count; i++)
        {
            colliders[i] = ScorepointParent.transform.GetChild(i).gameObject.name;
        }
        if (run)
        {
            

            cars = new List<CarControl>();
            redCars = initCars(redCarPrefab, redCarParent, redCarCount);
            greenCars = initCars(greenCarPrefab, greenCarParent, greenCarCount);
            blueCars = initCars(blueCarPrefab, blueCarParent, blueCarCount);
            orangeCars = initCars(orangeCarPrefab, orangeCarParent, orangeCarCount);
            yellowCar = initCars(yellowCarPrefab, yellowCarParent, 1)[0];
        }
    }

    public void resetAll()
    {
        for (int i = 0; i < redCars.Count; i++)
        {
            // generates a completely random set of weights
            redCars[i].nn.InitializeWeights();
        }

        for (int i = 0; i < greenCars.Count; i++)
        {
            // generates a completely random set of weights
            greenCars[i].nn.InitializeWeights();
        }

        for (int i = 0; i < blueCars.Count; i++)
        {
            // generates a completely random set of weights
            blueCars[i].nn.InitializeWeights();
        }

        for (int i = 0; i < orangeCars.Count; i++)
        {
            // generates a completely random set of weights
            orangeCars[i].nn.InitializeWeights();
        }

        yellowCar.nn.InitializeWeights();
        gen.text = "Generation: 0";
        score.text = "Score: 0";
        generation = 0;
        resetGame();
    }

    private List<CarControl> initCars(GameObject prefab, GameObject parent, int count)
    {
        List<CarControl> list = new List<CarControl>();
        for (int i = 0; i < count; i++)
        {
            CarControl car = Instantiate(prefab, parent.transform).GetComponent<CarControl>();
            list.Add(car);
            cars.Add(car);
        }
        return list;
    }
	
	// Update is called once per frame
	void Update () {
        if (run)
        {
            time += Time.deltaTime;

            bool active = false;
            for (int i = 0; i < cars.Count; i++)
            {
                if (cars[i].active)
                {
                    active = true;
                    break;
                }
            }

            if (!active || time > maxTime)
            { // end of this generation
                time = 0;
                int highestIndex = -1;
                int highestScore = 0;
                for (int i = 0; i < cars.Count; i++)
                {
                    if (cars[i].score >= highestScore)
                    {
                        highestIndex = i;
                        highestScore = cars[i].score;
                    }
                }

                double[] bestWeights = cars[highestIndex].nn.GetWeights();


                for (int i = 0; i < redCars.Count; i++)
                {
                    // generates a completely random set of weights
                    redCars[i].nn.InitializeWeights();
                }

                for (int i = 0; i < greenCars.Count; i++)
                {
                    // update weights towards best weights scaled by a factor
                    greenCars[i].nn.updateWeights(bestWeights, 80, 50);
                }

                for (int i = 0; i < blueCars.Count; i++)
                {
                    // update weights towards best weights scaled by a factor
                    blueCars[i].nn.updateWeights(bestWeights, 90, 20);
                }

                for (int i = 0; i < orangeCars.Count; i++)
                {
                    // copy of yellow car but with small mutation chance
                    orangeCars[i].nn.updateWeights(bestWeights, 100, 10);
                }

                // Yellow car will always have the best weights
                yellowCar.nn.SetWeights(bestWeights);

                //Debug.Log("Highest Score: " + highestScore + " Scored by " + cars[highestIndex].gameObject.name);

                score.text = "Score: " + highestScore;
                gen.text = "Generation: " + ++generation;
                resetGame();
            }
        }
    }

    private void printWeights(string s, double[] wgs)
    {
        for (int i = 0; i < wgs.Length; i++)
        {
            s += " " + i + ": " + wgs[i];
        }
        Debug.Log(s);
    }

    private void resetGame()
    {
        for (int i = 0; i < cars.Count; i++)
        {
            cars[i].resetGame();
        }
    }

    public void skipRound()
    {
        for (int i = 0; i < cars.Count; i++)
        {
            cars[i].active = false;
        }
    }

    public void LoadScene(string number)
    {
        SceneManager.LoadScene("Scene_"+number);
    }
}
