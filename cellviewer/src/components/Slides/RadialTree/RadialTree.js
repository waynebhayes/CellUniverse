import React, { Component } from 'react';
import { Container } from 'reactstrap';
import Slider from '@material-ui/lab/Slider';
// import Cell from "./Cell/Cell.js"
import './RadialTree.css';


const data = [
    {title: "", value: 1, color: "rgb(255,0,0)"},
    {title: "", value: 1, color: "rgb(0,255,0)"},
    {title: "", value: 1, color: "rgb(0,0,255)"},
    {title: "", value: 1, color: "rgb(255,255,0)"}
  ]
  

export default class RadialTree extends Component {
    constructor(props) {
        super(props);
        this.changeColor = this.changeColor.bind(this);
        this.state = {
            posC: 0
        }
    }

    changeColor(e,i){
        this.setState({
            posC : i
        });
    }

    render() {

        var posC = this.state.posC;
        return (
            <Container 
                    style={{
                        width:"50vh",
                        position:"relative",
                        margin:"0", 
                        padding:"0",
                        display:"contents"}}>
                <img
                    alt=""
                    style={{
                        position:"absolute",
                        height: "79vh",
                        width: "39vw",
                        right:"0.5vw",
                        top:"0.5vh",
                        margin: "10vh",
                        marginLeft: "5vw",
                        marginRight: "5vw",
                        backgroundColor:"gray",
                        display:["none","initial"][posC]
                    }}
                />
                <img
                    id="pie"
                    src={this.props.src_pie}
                    alt="im"
                    className="image2"
                    style={{
                        height: (this.props.pos*80).toString()+"vh",
                        width: (this.props.pos*40).toString()+"vw",
                        right: ((40-this.props.pos*40)/2).toString()+"vw",
                        top: ((80-this.props.pos*80)/2).toString()+"vh"
                    }}/>
                <img
                    id="tree"
                    src={this.props.src} 
                    alt="im"
                    className="image2"
                    style={{
                        height: "80vh",
                        width: "40vw",
                        right:"0",
                        top:"0"
                    }}
                    />

                <Slider
                    value={posC}
                    min={0}
                    max={1}
                    step={1}
                    onChange={this.changeColor}
                    onClick={()=>{this.changeColor(null,1-posC)}}
                    style={{
                        width:"2%",
                        height:"5%",
                        position:"absolute",
                        top: "5%",
                        right: "10%"
                    }}
                />
                
            </Container>
        );
    }
}
