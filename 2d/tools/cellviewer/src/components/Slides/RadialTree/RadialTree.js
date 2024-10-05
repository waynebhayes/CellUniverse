import React, { Component } from 'react';
import { Container } from 'reactstrap';
import Switch from '@material-ui/core/Switch';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Canvas from './Canvas.js'

const bool_int = {"false":0, "true": 1}

export default class RadialTree extends Component {
    constructor(props) {
        super(props);
        this.changeColor = this.changeColor.bind(this);
        this.state = {
            posC: false
        }
    }

    changeColor(e,i){
        console.log(e,i);
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

                <Canvas 
                    angles={this.props.angles}
                    frames={this.props.frames}
                    colors={this.props.colors}
                    curr={this.props.curr}
                    l={l}
                    posC={posC}
                    style={{
                        position:"absolute",
                        right: 0,
                        top: 0
                    }}/>

                <Typography component="div" style={{
                        position:"absolute",
                        opacity: "0.5",
                        borderRadius: "20px",
                        top: "2%",
                        right: "10%",
                        color: "white",
                        backgroundColor:"#555555",
                        paddingLeft: "2%",
                        paddingRight: "2%",
                        paddingTop: "0.5%",
                        paddingBottom: "0.5%"
                    }}>
                    <Grid component="label" container alignItems="center"
                            style={{
                            }}>
                        <Grid item>Show Path</Grid>
                        <Grid item>
                            <Switch
                                color="primary"
                                checked={posC}
                                onChange={this.changeColor}
                                value={posC}
                            />
                        </Grid>
                    </Grid>
                </Typography>
                
            </Container>
        );
    }
}
