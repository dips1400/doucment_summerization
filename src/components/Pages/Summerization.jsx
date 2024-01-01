import React from 'react'
import './Summerization.css'
import { useState, useEffect} from 'react';

const Summerization = () => {

  const [result, setResult] = useState()
  const [question, setQuestion] = useState()
  const [file, setFile] = useState()


  const handleQuestionChange = (event) => {
    setQuestion(event.target.value)
  }

  const handleFileChange = (event) => {
    setFile(event.target.files[0])
  }

  const handleSubmit = (event) => {
    event.preventDefault()

    const formData = new FormData()

    if (file) {
      formData.append('file', file)
      console.log('got file')
    }
    if (question) {
      formData.append('question', question)
      console.log('got question')
    }

    fetch('/getdata', {
      method: "POST",
      formData
    })
      .then((response) => response.json())
      .then((data) => {
        setResult(data.result)
        console.log(data)
      })
      .catch((error) => {
      console.error("Error", error)
    })
  
  }


  return (
    <div>
      <div className="container">
        <div className='header-container'>
            <p>Summerization your document</p>
        </div>
        <div className='main-content'>
            <div className='form-container'>
                <form 
                // onSubmit={handleSubmit} 
                action='/getdata'
                className="form-content" 
                method="post" encType="multipart/form-data">
                  <div className='inner-container'>

                  {/* <div className='csv_file'>
                    <label name="file" htmlFor="file">
                      Upload CSV file:
                    </label>
                    <br></br><br></br>
                    <input
                      type="file"
                      id="file"
                      name="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      className="mb-3"
                    />
                    </div> */}

                    <div className='question_div'> 
                      <label htmlFor="question">
                        Get summary of the file(text_segments.csv):
                      </label>
                      {/* <input
                      name="question"
                      id="question"
                      type="text"
                      value={question}
                      onChange={handleQuestionChange}
                      placeholder="Ask your question here"
                    /> */}
                    <button
                    name='submit'
                    type="submit"
                    // disabled={!file}
                  >
                    Generate
                  </button>
                    </div>
                  
                  </div>
                </form>
                <p>Result: 
                  {result}
                </p>
              </div>
          </div>
      </div>
    </div>
  )
}

export default Summerization