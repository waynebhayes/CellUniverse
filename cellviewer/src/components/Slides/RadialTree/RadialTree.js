import React, { Component } from 'react';
import { Container } from 'reactstrap';
import Slider from '@material-ui/lab/Slider';

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

        var l = window.innerHeight
        if(window.innerHeight*2 > window.innerWidth){
            l = window.innerWidth/2
        }

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
                        height: (l*.79).toString()+"px",
                        width: (l*.79).toString()+"px",
                        right: (window.innerWidth/4-(l*.79/2)).toString()+"px",
                        top: (window.innerHeight/2-(l*.79/2)).toString()+"px",
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
                        position:"absolute",
                        height: (this.props.pos*l*.8).toString()+"px",
                        width: (this.props.pos*l*.8).toString()+"px",
                        right: (window.innerWidth/4-this.props.pos*l*.4).toString()+"px",
                        top: (window.innerHeight/2-this.props.pos*l*.4).toString()+"px"
                    }}/>
                <img
                    id="tree"
                    src={this.props.src} 
                    alt="im"
                    className="image2"
                    style={{
                        position:"absolute",
                        height: (l*.8).toString()+"px",
                        width: (l*.8).toString()+"px",
                        right: (window.innerWidth/4-l*.4).toString()+"px",
                        top: (window.innerHeight/2-l*.4).toString()+"px"
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
