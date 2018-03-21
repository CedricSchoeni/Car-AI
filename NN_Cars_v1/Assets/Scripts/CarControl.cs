using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class CarControl : MonoBehaviour {

    [SerializeField]
    public bool run = true;
    public float velocity;
    public float left90;
    public float left45;
    public float fd;
    public float right45;
    public float right90;
    private float accelerationChange = 0;
    private float rotationChange = 0;
    public DeepNeuralNetwork nn;

    

    LayerMask lm;

    public int score = 0;
    public bool active = true;

    private int colliderIndex = 0;

    // Use this for initialization
    void Start () {
        nn = new DeepNeuralNetwork(6, new int[] { 5, 4 }, 4);
        lm = 1 << 8;
    }

    public void resetGame()
    {
        score = 0;
        active = true;
        accelerationChange = 0;
        rotationChange = 0;
        velocity = 0;
        colliderIndex = 0;
        resetPosition();
    }

    private void resetPosition()
    {
        transform.position = new Vector3(0, 0, 0);
        transform.rotation = Quaternion.Euler(new Vector3(0, 0, 0));
    }

    public void moveCar(int output)
    {
        switch (output)
        {
            // Turn Right
            case 0:
                rotationChange = 90;
                accelerationChange = 0;
                break;
            // Turn Left
            case 1:
                rotationChange = -90;
                accelerationChange = 0;
                break;
            // Accelerate
            case 2:
                rotationChange = 0;
                accelerationChange = 0.1f;
                break;
            // Stop
            case 3:
                rotationChange = 0;
                accelerationChange = -0.1f;
                break;
            default:
                break;
        }
    }
	
	// Update is called once per frame
	void Update () {
        if (active)
        {
            // apply accelerationChange to velocity of the car
            velocity += accelerationChange;

            transform.Translate(velocity * Time.deltaTime, 0, 0);
            transform.Rotate(0, rotationChange * Time.deltaTime, 0);

            Vector3 left_90 = transform.TransformDirection(Vector3.forward);
            Vector3 left_45 = transform.TransformDirection(Vector3.right - Vector3.back);
            Vector3 forward = transform.TransformDirection(Vector3.right);
            Vector3 right_45= transform.TransformDirection(Vector3.right - Vector3.forward);
            Vector3 right_90 = transform.TransformDirection(Vector3.back);

            Debug.DrawRay(transform.position, left_90 * 6, Color.red);
            Debug.DrawRay(transform.position, left_45 * 4.243f, Color.yellow);
            Debug.DrawRay(transform.position, forward * 6, Color.green);
            Debug.DrawRay(transform.position, right_45 * 4.243f, Color.blue);
            Debug.DrawRay(transform.position, right_90 * 6, Color.magenta);
            RaycastHit hit;

               
            left90 = 6;
            left45 = 6;
            fd = 6;
            right45 = 6;
            right90 = 6;

            if (Physics.Raycast(transform.position, left_90, out hit, 6, lm.value))
                if (hit.transform.gameObject.name == "Cube")
                {
                    left90 = hit.distance - 0.5f;
                }
                    
            if (Physics.Raycast(transform.position, left_45, out hit, 6, lm.value))
                if (hit.transform.gameObject.name == "Cube")
                {
                    left45 = hit.distance - 0.707f;
                }
                    
            if (Physics.Raycast(transform.position, forward, out hit, 6, lm.value))
                if (hit.transform.gameObject.name == "Cube")
                {
                    fd = hit.distance - 1f;
                }
                    
            if (Physics.Raycast(transform.position, right_45, out hit, 6, lm.value))
                if (hit.transform.gameObject.name == "Cube")
                {
                    right45 = hit.distance - 0.707f;
                }
                    
            if (Physics.Raycast(transform.position, right_90, out hit, 6, lm.value))
                if (hit.transform.gameObject.name == "Cube")
                {
                    right90 = hit.distance - 0.5f;
                }
                    


            double[] output = nn.ComputeOutputs(new double[] { left90, left45, fd, right45, right90, velocity });
            double maxValue = output.Max();
            int maxIndex = output.ToList().IndexOf(maxValue);
            if (run) moveCar(maxIndex);
        }
    }

    void OnTriggerEnter(Collider col)
    {
        if (col.transform.gameObject.name == "Cube")
        {
            if (run) active = false;
        }
        /** old fitness function - not working properly */
        if (col.transform.gameObject.name == CarController.colliders[colliderIndex])
        {



            float distance = Mathf.Max(Vector3.Distance(col.transform.position, transform.position), 1);
            score += (int) (8 / distance + 5);
            colliderIndex++;
            if (colliderIndex == CarController.colliders.Length)
            {
                colliderIndex = 0;
            }
        }
        
}
}
